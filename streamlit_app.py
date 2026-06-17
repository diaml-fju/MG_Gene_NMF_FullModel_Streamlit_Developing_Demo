from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PICKLE_DIR = APP_DIR / "pickle_File"
ASV_FAMILY_FILE =  "ASV_Family_name.csv"
USER_GUIDE_PDF = APP_DIR / "User guide for GutMGS Final.pdf"
SHIFT_VALUE = 14
MANUAL_CASE_LABEL = "Manual input"
FT_NAME_MAP = {
    "FT0047": "Racemethionine",
    "FT0270": "Ergocalciferol",
    "FT0131": "3-HAA",
    "FT2032": "Deoxyuridine",
    "FT1558": "benzothiazole-2(3H)-thione",
}
PHYLUM_ORDER = [
    "Firmicutes",
    "Bacteroidota",
    "Proteobacteria",
    "Actinobacteriota",
    "Patescibacteria",
    "Fusobacteriota",
    "Cyanobacteria",
    "Desulfobacterota",
    "Spirochaetota",
    "Unassigned",
]
FAMILY_PHYLUM_MAP = {
    "Acidaminococcaceae": "Firmicutes",
    "Actinomycetaceae": "Actinobacteriota",
    "Aerococcaceae": "Firmicutes",
    "Anaerovoracaceae": "Firmicutes",
    "Atopobiaceae": "Actinobacteriota",
    "Bacteroidaceae": "Bacteroidota",
    "Barnesiellaceae": "Bacteroidota",
    "Burkholderiaceae": "Proteobacteria",
    "Butyricicoccaceae": "Firmicutes",
    "Chloroplast": "Cyanobacteria",
    "Christensenellaceae": "Firmicutes",
    "Clostridia_UCG-014": "Firmicutes",
    "Clostridiaceae": "Firmicutes",
    "Desulfovibrionaceae": "Desulfobacterota",
    "Eggerthellaceae": "Actinobacteriota",
    "Enterobacteriaceae": "Proteobacteria",
    "Erysipelatoclostridiaceae": "Firmicutes",
    "Erysipelotrichaceae": "Firmicutes",
    "Fusobacteriaceae": "Fusobacteriota",
    "Lachnospiraceae": "Firmicutes",
    "Lactobacillaceae": "Firmicutes",
    "Marinifilaceae": "Bacteroidota",
    "Oxalobacteraceae": "Proteobacteria",
    "Pasteurellaceae": "Proteobacteria",
    "Prevotellaceae": "Bacteroidota",
    "Rikenellaceae": "Bacteroidota",
    "Ruminococcaceae": "Firmicutes",
    "Saccharimonadaceae": "Patescibacteria",
    "Saccharimonadales": "Patescibacteria",
    "Spirochaetaceae": "Spirochaetota",
    "Streptococcaceae": "Firmicutes",
    "Sutterellaceae": "Proteobacteria",
    "Tannerellaceae": "Bacteroidota",
    "Veillonellaceae": "Firmicutes",
    "Others": "Unassigned",
}


def resolve_asset_path(filename: str) -> Path:
    for base_dir in (PICKLE_DIR, APP_DIR):
        path = base_dir / filename
        if path.exists():
            return path
    return PICKLE_DIR / filename


@st.cache_resource(show_spinner="Loading model assets...")
def load_model_assets() -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(resolve_asset_path("MG_Gene_NMF_Core.pkl"), "rb") as file:
            core_elements = pickle.load(file)
        with open(resolve_asset_path("MG_Gene_NMF_Reference.pkl"), "rb") as file:
            reference_data = pickle.load(file)
        with open(resolve_asset_path("MG_Gene_NMF_Example_Positive_Cases.pkl"), "rb") as file:
            demo_positive = pickle.load(file)
        with open(resolve_asset_path("MG_Gene_NMF_Example_Negative_Cases.pkl"), "rb") as file:
            demo_negative = pickle.load(file)

    nmf_model = core_elements["NMF"]
    if not hasattr(nmf_model, "alpha_W"):
        nmf_model.alpha_W = getattr(nmf_model, "alpha", 0.0)
    if not hasattr(nmf_model, "alpha_H"):
        nmf_model.alpha_H = getattr(nmf_model, "alpha", 0.0)

    xgb_model = core_elements["XGBModel"]
    if getattr(xgb_model, "missing", None) is None:
        xgb_model.missing = np.nan

    demo_cases = {
        **{f"Positive - {name}": case for name, case in demo_positive.items()},
        **{f"Negative - {name}": case for name, case in demo_negative.items()},
    }
    return core_elements, reference_data, demo_cases


@st.cache_data(show_spinner=False)
def load_asv_family_info() -> tuple[dict[str, str], dict[str, int]]:
    if not ASV_FAMILY_FILE.exists():
        return {}, {}

    family_df = pd.read_csv(ASV_FAMILY_FILE)
    if not {"Var", "Family"}.issubset(family_df.columns):
        return {}, {}

    family_df = family_df.dropna(subset=["Var", "Family"])
    family_df["Var"] = family_df["Var"].astype(str)
    family_df["Family"] = family_df["Family"].astype(str)

    family_map = dict(zip(family_df["Var"], family_df["Family"]))
    family_counts = family_df["Family"].value_counts().to_dict()
    phylum_rank = {phylum: order for order, phylum in enumerate(PHYLUM_ORDER)}

    def sort_key(family: str) -> tuple[int, int, str]:
        clean_family = clean_family_name(family)
        phylum = FAMILY_PHYLUM_MAP.get(clean_family, "Unassigned")
        return (
            phylum_rank.get(phylum, len(PHYLUM_ORDER)),
            -int(family_counts.get(family, 0)),
            clean_family,
        )

    ordered_families = sorted(family_df["Family"].drop_duplicates().tolist(), key=sort_key)
    family_order = {family: order for order, family in enumerate(ordered_families)}
    return family_map, family_order


def update_templates(
    before_asv: pd.DataFrame,
    after_asv: pd.DataFrame,
    before_ft: pd.DataFrame,
    after_ft: pd.DataFrame,
    user_input: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    before_asv = before_asv.copy()
    after_asv = after_asv.copy()
    before_ft = before_ft.copy()
    after_ft = after_ft.copy()

    for feature, values in user_input.items():
        before_value = float(values.get("before", 0) or 0)
        after_value = float(values.get("after", 0) or 0)

        if feature in before_asv.columns:
            before_asv.loc[0, feature] = before_value
            after_asv.loc[0, feature] = after_value
        elif feature in before_ft.columns:
            before_ft.loc[0, feature] = before_value
            after_ft.loc[0, feature] = after_value

    return before_asv, after_asv, before_ft, after_ft


def clr_transform(df: pd.DataFrame | pd.Series, pseudo_count: float) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    else:
        df = pd.DataFrame(df).copy()

    df = df.astype(float).replace(0, pseudo_count)
    df_prop = df.div(df.sum(axis=1), axis=0)
    log_df = np.log(df_prop)
    clr_df = log_df.sub(log_df.mean(axis=1), axis=0)
    clr_df.columns = df.columns
    clr_df.index = df.index
    return clr_df


def preprocess_new_subject(
    before_asv: pd.DataFrame,
    after_asv: pd.DataFrame,
    before_ft: pd.DataFrame,
    after_ft: pd.DataFrame,
    pseudo_count: float,
) -> pd.DataFrame:
    before_asv_clr = clr_transform(before_asv, pseudo_count)
    after_asv_clr = clr_transform(after_asv, pseudo_count)
    asv_diff = after_asv_clr - before_asv_clr

    before_ft = before_ft.copy().astype(float)
    after_ft = after_ft.copy().astype(float)
    ft_relative = pd.DataFrame(index=before_ft.index, columns=before_ft.columns, dtype=float)
    row_idx = before_ft.index[0]

    for col in before_ft.columns:
        before = before_ft.loc[row_idx, col]
        after = after_ft.loc[row_idx, col]

        if before <= 0 and after <= 0:
            ft_relative.loc[row_idx, col] = 0
        elif before <= 0:
            ft_relative.loc[row_idx, col] = after / pseudo_count
        elif after <= 0:
            ft_relative.loc[row_idx, col] = pseudo_count / before
        else:
            ft_relative.loc[row_idx, col] = after / before

    return pd.concat([asv_diff, ft_relative], axis=1)


def case_dict_to_table(
    case: dict[str, dict[str, float]],
    selected_features: list[str],
    reference_data: dict[str, pd.DataFrame],
    asv_family_map: dict[str, str],
    asv_family_order: dict[str, int],
) -> pd.DataFrame:
    asv_features = set(reference_data["ASV_Before_Ref"].columns)
    rows = []

    for feature in selected_features:
        values = case.get(feature, {})
        feature_type = "ASV" if feature in asv_features else "FT"
        family = asv_family_map.get(feature, "Unmapped family") if feature_type == "ASV" else "FT features"
        rows.append(
            {
                "feature": feature,
                "type": feature_type,
                "family": family,
                "family_order": asv_family_order.get(family, 9999) if feature_type == "ASV" else 10000,
                "display_name": FT_NAME_MAP.get(feature, "") if feature_type == "FT" else "",
                "before": float(values.get("before", 0) or 0),
                "after": float(values.get("after", 0) or 0),
            }
        )

    return pd.DataFrame(rows)


def table_to_case(input_table: pd.DataFrame) -> dict[str, dict[str, float]]:
    case = {}
    for row in input_table.itertuples(index=False):
        case[str(row.feature)] = {
            "before": float(row.before or 0),
            "after": float(row.after or 0),
        }
    return case


def input_widget_key(case_label: str, feature: str, field: str) -> str:
    safe_case_label = case_label.replace(" ", "_").replace("-", "_")
    return f"case_input_{safe_case_label}_{feature}_{field}"


def empty_case(selected_features: list[str]) -> dict[str, dict[str, float]]:
    return {feature: {"before": 0.0, "after": 0.0} for feature in selected_features}


def case_values_changed(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_values = left[["feature", "before", "after"]].sort_values("feature").reset_index(drop=True)
    right_values = right[["feature", "before", "after"]].sort_values("feature").reset_index(drop=True)
    if not left_values["feature"].equals(right_values["feature"]):
        return True
    return not np.allclose(
        left_values[["before", "after"]].to_numpy(dtype=float),
        right_values[["before", "after"]].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )


def build_feature_groups(input_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {}

    asv_table = input_table[input_table["type"] == "ASV"].sort_values(["family_order", "feature"])
    for family, family_table in asv_table.groupby("family", sort=False):
        display_family = format_family_with_count(str(family), len(family_table))
        groups[f"ASV - {display_family}"] = family_table

    ft_table = input_table[input_table["type"] != "ASV"]
    if not ft_table.empty:
        groups["FT features"] = ft_table

    return groups


def collect_sidebar_case(input_table: pd.DataFrame, selected_case: str, on_value_change: Any | None = None) -> pd.DataFrame:
    edited_rows = []
    feature_groups = build_feature_groups(input_table)

    for group_name, group_table in feature_groups.items():
        non_zero_count = int(((group_table["before"] != 0) | (group_table["after"] != 0)).sum())
        if group_name.startswith("ASV - "):
            label = group_name
        else:
            label = f"{group_name} ({len(group_table)} features"
        if non_zero_count:
            if group_name.startswith("ASV - "):
                label += f" ({non_zero_count} filled)"
            else:
                label += f", {non_zero_count} filled"
        if not group_name.startswith("ASV - "):
            label += ")"

        with st.sidebar.expander(label, expanded=False):
            for row in group_table.itertuples(index=False):
                caption = str(row.feature)
                if row.type == "FT" and row.display_name:
                    caption = f"{row.feature} - {row.display_name}"
                st.caption(caption)
                before_col, after_col = st.columns(2)
                before_value = before_col.number_input(
                    "Before",
                    min_value=0.0,
                    value=float(row.before),
                    format="%.8f",
                    key=input_widget_key(selected_case, row.feature, "before"),
                    on_change=on_value_change,
                    args=(selected_case, row.feature, "before") if on_value_change else None,
                )
                after_value = after_col.number_input(
                    "After",
                    min_value=0.0,
                    value=float(row.after),
                    format="%.8f",
                    key=input_widget_key(selected_case, row.feature, "after"),
                    on_change=on_value_change,
                    args=(selected_case, row.feature, "after") if on_value_change else None,
                )
                edited_rows.append(
                    {
                        "feature": row.feature,
                        "type": row.type,
                        "family": row.family,
                        "family_order": row.family_order,
                        "display_name": row.display_name,
                        "before": before_value,
                        "after": after_value,
                    }
                )

    edited_table = pd.DataFrame(edited_rows)
    feature_order = {feature: order for order, feature in enumerate(input_table["feature"])}
    edited_table["feature_order"] = edited_table["feature"].map(feature_order)
    edited_table = edited_table.sort_values("feature_order").drop(columns=["feature_order"])
    return edited_table.reset_index(drop=True)


def build_input_change_table(input_table: pd.DataFrame, include_zero_features: bool = False) -> pd.DataFrame:
    change_table = input_table.copy()
    change_table["label"] = change_table["feature"]
    ft_mask = (change_table["type"] == "FT") & (change_table["display_name"] != "")
    change_table.loc[ft_mask, "label"] = (
        change_table.loc[ft_mask, "feature"] + " - " + change_table.loc[ft_mask, "display_name"]
    )
    change_table["delta"] = change_table["after"] - change_table["before"]

    if not include_zero_features:
        change_table = change_table[(change_table["before"] != 0) | (change_table["after"] != 0)]

    return change_table[["label", "type", "family", "before", "after", "delta"]]


def build_ft_total_table(input_table: pd.DataFrame) -> pd.DataFrame:
    ft_table = input_table[input_table["type"] == "FT"].copy()
    if ft_table.empty:
        return pd.DataFrame(columns=["Feature", "Compound", "Before", "After"])

    ft_table["Compound"] = np.where(ft_table["display_name"] != "", ft_table["display_name"], ft_table["feature"])
    return ft_table.rename(columns={"feature": "Feature", "before": "Before", "after": "After"})[
        ["Feature", "Compound", "Before", "After"]
    ]


def build_ft_change_table(input_table: pd.DataFrame) -> pd.DataFrame:
    ft_table = build_ft_total_table(input_table)
    if ft_table.empty:
        return pd.DataFrame(columns=["Feature", "Compound", "Before", "After", "After - Before"])

    ft_table["After - Before"] = ft_table["After"] - ft_table["Before"]
    return ft_table


def clean_family_name(family: str) -> str:
    return str(family).removeprefix("f__")


def format_family_with_count(family: str, count: int | float) -> str:
    return f"{clean_family_name(family)} (n={int(count)})"


def build_asv_family_clr_change_table(
    input_table: pd.DataFrame,
    pseudo_count: float,
    include_zero_families: bool = False,
) -> pd.DataFrame:
    asv_table = input_table[input_table["type"] == "ASV"].copy()
    if asv_table.empty:
        return pd.DataFrame(
            columns=["Family", "Before total", "After total", "Before CLR", "After CLR", "CLR delta"]
        )

    family_totals = (
        asv_table.groupby("family", sort=False)
        .agg(
            before=("before", "sum"),
            after=("after", "sum"),
            family_order=("family_order", "min"),
            asv_count=("feature", "count"),
        )
        .rename_axis("family")
        .reset_index()
        .sort_values("family_order")
    )
    family_totals["Family"] = family_totals.apply(
        lambda row: format_family_with_count(row["family"], row["asv_count"]),
        axis=1,
    )

    before_family = pd.DataFrame([family_totals["before"].to_numpy()], columns=family_totals["Family"])
    after_family = pd.DataFrame([family_totals["after"].to_numpy()], columns=family_totals["Family"])
    before_clr = clr_transform(before_family, pseudo_count)
    after_clr = clr_transform(after_family, pseudo_count)

    family_totals["Before CLR"] = before_clr.iloc[0].to_numpy()
    family_totals["After CLR"] = after_clr.iloc[0].to_numpy()
    family_totals["CLR delta"] = family_totals["After CLR"] - family_totals["Before CLR"]

    if not include_zero_families:
        family_totals = family_totals[(family_totals["before"] != 0) | (family_totals["after"] != 0)]

    return family_totals.rename(
        columns={
            "before": "Before total",
            "after": "After total",
        }
    )[["Family", "family_order", "Before total", "After total", "Before CLR", "After CLR", "CLR delta"]]


def build_asv_family_clr_value_table(
    input_table: pd.DataFrame,
    pseudo_count: float,
) -> pd.DataFrame:
    clr_table = build_asv_family_clr_change_table(
        input_table,
        pseudo_count,
        include_zero_families=True,
    )
    if clr_table.empty:
        return pd.DataFrame(columns=["Family", "family_order", "Before CLR", "After CLR"])

    return clr_table[["Family", "family_order", "Before CLR", "After CLR"]]


def build_asv_family_total_table(input_table: pd.DataFrame) -> pd.DataFrame:
    asv_table = input_table[input_table["type"] == "ASV"].copy()
    if asv_table.empty:
        return pd.DataFrame(columns=["Family", "family_order", "Before total", "After total"])

    family_totals = (
        asv_table.groupby("family", sort=False)
        .agg(
            before=("before", "sum"),
            after=("after", "sum"),
            family_order=("family_order", "min"),
            asv_count=("feature", "count"),
        )
        .rename_axis("family")
        .reset_index()
        .sort_values("family_order")
    )
    family_totals["Family"] = family_totals.apply(
        lambda row: format_family_with_count(row["family"], row["asv_count"]),
        axis=1,
    )

    return family_totals.rename(
        columns={
            "before": "Before total",
            "after": "After total",
        }
    )[["Family", "family_order", "Before total", "After total"]]


def predict_case(
    user_input: dict[str, dict[str, float]],
    core_elements: dict[str, Any],
    reference_data: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    before_asv, after_asv, before_ft, after_ft = update_templates(
        reference_data["ASV_Before_Ref"],
        reference_data["ASV_After_Ref"],
        reference_data["FT_Before_Ref"],
        reference_data["FT_After_Ref"],
        user_input,
    )

    x_new = preprocess_new_subject(
        before_asv,
        after_asv,
        before_ft,
        after_ft,
        float(core_elements["ImputeValue"]),
    )
    x_new = x_new + SHIFT_VALUE

    normalized = core_elements["MMScaler"].transform(x_new.to_numpy())
    normalized_df = pd.DataFrame(np.clip(normalized, 0, 1), columns=x_new.columns)

    w_new = core_elements["NMF"].transform(normalized_df.to_numpy())
    w_new_df = pd.DataFrame(w_new)
    selected_components = core_elements["Sel_W_Components"]
    w_new_final = w_new_df.iloc[:, selected_components].copy()

    h_final = core_elements["H_Final"]
    new_x_recon = pd.DataFrame(np.dot(w_new_final, h_final), columns=h_final.columns)

    threshold = float(core_elements["UserTH"])
    probability = float(core_elements["XGBModel"].predict_proba(new_x_recon)[:, 1][0])
    prediction = int(probability >= threshold)

    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": prediction,
        "label": "Improve (Positive)" if prediction == 1 else "Non-Improve (Negative)",
        "processed_features": x_new,
        "reconstructed_features": new_x_recon,
        "nmf_components": w_new_final,
    }


def render_user_guide_download() -> None:
    if USER_GUIDE_PDF.exists():
        st.download_button(
            "Download USER GUIDE",
            data=USER_GUIDE_PDF.read_bytes(),
            file_name="USER GUIDE.pdf",
            mime="application/pdf",
        )
    else:
        st.info("USER GUIDE PDF is not available yet.")


def render_prediction_summary(prediction_result: dict[str, Any] | None, prediction_error: Exception | None) -> None:
    if prediction_error is not None:
        st.error("Prediction failed. Please check the input values.")
        st.exception(prediction_error)
        return

    if prediction_result is None:
        st.info("Fill the values in the sidebar, then press Run prediction.")
        return

    st.metric("Prediction", prediction_result["label"])
    if prediction_result.get("forced_no_input"):
        st.warning("No input values were provided, so this case is forced to Non-Improve (Negative).")
    elif int(prediction_result["prediction"]) == 1:
        st.success("This case is predicted as Improve (Positive).")
    else:
        st.warning("This case is predicted as Non-Improve (Negative).")


def render_family_clr_values_section(
    edited_input: pd.DataFrame,
    pseudo_count: float,
    *,
    show_table_expander: bool = True,
) -> None:
    st.subheader("ASV Family Values")
    st.caption("Family ASV values are summed first, then Before and After are CLR-transformed separately.")
    family_clr_value_table = build_asv_family_clr_value_table(edited_input, pseudo_count)

    if family_clr_value_table.empty:
        st.info("No ASV family CLR values to plot yet.")
        return

    family_clr_value_long = family_clr_value_table.melt(
        id_vars=["Family", "family_order"],
        value_vars=["Before CLR", "After CLR"],
        var_name="Timepoint",
        value_name="CLR value",
    )
    family_clr_value_long["Timepoint"] = family_clr_value_long["Timepoint"].str.replace(" CLR", "", regex=False)
    family_clr_value_long["Timepoint order"] = family_clr_value_long["Timepoint"].map({"Before": 0, "After": 1})
    clr_value_height = max(420, min(1000, len(family_clr_value_table) * 42))

    family_clr_value_chart = (
        alt.Chart(family_clr_value_long)
        .mark_bar()
        .encode(
            x=alt.X("CLR value:Q", title="CLR value"),
            y=alt.Y(
                "Family:N",
                title=None,
                axis=alt.Axis(labelLimit=320),
                sort=alt.EncodingSortField(field="family_order", order="ascending"),
            ),
            yOffset=alt.YOffset(
                "Timepoint:N",
                sort=alt.EncodingSortField(field="Timepoint order", order="ascending"),
            ),
            color=alt.Color(
                "Timepoint:N",
                scale=alt.Scale(domain=["Before", "After"], range=["#2f6fbb", "#c94c4c"]),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Family", "Timepoint", "CLR value"],
        )
        .properties(height=clr_value_height)
    )

    st.altair_chart(family_clr_value_chart, use_container_width=True)
    table = family_clr_value_table.drop(columns=["family_order"])
    if show_table_expander:
        with st.expander("Show ASV family values table", expanded=False):
            st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_family_clr_change_section(
    edited_input: pd.DataFrame,
    pseudo_count: float,
    *,
    show_table_expander: bool = True,
) -> None:
    st.subheader("Change")
    family_clr_table = build_asv_family_clr_change_table(
        edited_input,
        pseudo_count,
        include_zero_families=True,
    )

    if family_clr_table.empty:
        st.info("No ASV family values to plot yet.")
        return

    family_chart_data = family_clr_table.copy()
    family_chart_data["Direction"] = np.where(family_chart_data["CLR delta"] >= 0, "Increase", "Decrease")
    family_chart_data["Zero"] = 0
    family_chart_height = max(320, min(900, len(family_chart_data) * 30))
    max_abs_clr_delta = float(family_chart_data["CLR delta"].abs().max() or 1)

    family_bars = (
        alt.Chart(family_chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "CLR delta:Q",
                title="CLR(After) - CLR(Before)",
                axis=alt.Axis(grid=True),
                scale=alt.Scale(domain=[-max_abs_clr_delta, max_abs_clr_delta]),
            ),
            y=alt.Y(
                "Family:N",
                title=None,
                axis=alt.Axis(labelLimit=320),
                sort=alt.EncodingSortField(field="family_order", order="ascending"),
            ),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Increase", "Decrease"], range=["#2f7d5c", "#b34d4d"]),
                legend=None,
            ),
            tooltip=["Family", "Before total", "After total", "Before CLR", "After CLR", "CLR delta"],
        )
        .properties(height=family_chart_height)
    )

    family_zero_rule = (
        alt.Chart(family_chart_data)
        .mark_rule(color="#444444", strokeWidth=2)
        .encode(x="Zero:Q")
    )

    st.altair_chart(family_zero_rule + family_bars, use_container_width=True)
    table = family_chart_data.drop(columns=["Direction", "family_order", "Zero"])
    if show_table_expander:
        with st.expander("Show change table", expanded=False):
            st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_ft_values_section(
    edited_input: pd.DataFrame,
    *,
    show_table_expander: bool = True,
) -> None:
    st.subheader("FT Values")
    st.caption("The five selected TCM/FT features are shown separately from ASV families.")
    ft_total_table = build_ft_total_table(edited_input)

    if ft_total_table.empty:
        st.info("No FT values to plot yet.")
        return

    ft_long = ft_total_table.melt(
        id_vars=["Feature", "Compound"],
        value_vars=["Before", "After"],
        var_name="Timepoint",
        value_name="Value",
    )
    ft_long["Label"] = ft_long["Feature"] + " - " + ft_long["Compound"]
    ft_long["Feature order"] = ft_long["Feature"].map(
        {feature: order for order, feature in enumerate(ft_total_table["Feature"].tolist())}
    )
    ft_long["Timepoint order"] = ft_long["Timepoint"].map({"Before": 0, "After": 1})
    ft_height = max(260, len(ft_total_table) * 56)

    ft_chart = (
        alt.Chart(ft_long)
        .mark_bar()
        .encode(
            x=alt.X("Value:Q", title="Value"),
            y=alt.Y(
                "Label:N",
                title=None,
                axis=alt.Axis(labelLimit=360),
                sort=alt.EncodingSortField(field="Feature order", order="ascending"),
            ),
            yOffset=alt.YOffset(
                "Timepoint:N",
                sort=alt.EncodingSortField(field="Timepoint order", order="ascending"),
            ),
            color=alt.Color(
                "Timepoint:N",
                scale=alt.Scale(domain=["Before", "After"], range=["#2f6fbb", "#c94c4c"]),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Feature", "Compound", "Timepoint", "Value"],
        )
        .properties(height=ft_height)
    )

    st.altair_chart(ft_chart, use_container_width=True)
    if show_table_expander:
        with st.expander("Show FT values table", expanded=False):
            st.dataframe(ft_total_table, use_container_width=True, hide_index=True)
    else:
        st.dataframe(ft_total_table, use_container_width=True, hide_index=True)


def render_ft_change_section(
    edited_input: pd.DataFrame,
    *,
    show_table_expander: bool = True,
) -> None:
    st.subheader("FT Change")
    ft_change_table = build_ft_change_table(edited_input)

    if ft_change_table.empty:
        st.info("No FT changes to plot yet.")
        return

    chart_data = ft_change_table.copy()
    chart_data["Label"] = chart_data["Feature"] + " - " + chart_data["Compound"]
    chart_data["Direction"] = np.where(chart_data["After - Before"] >= 0, "Increase", "Decrease")
    chart_data["Feature order"] = chart_data["Feature"].map(
        {feature: order for order, feature in enumerate(ft_change_table["Feature"].tolist())}
    )
    chart_data["Zero"] = 0
    max_abs_delta = float(chart_data["After - Before"].abs().max() or 1)
    ft_change_height = max(260, len(chart_data) * 54)

    bars = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "After - Before:Q",
                title="After - Before",
                axis=alt.Axis(grid=True),
                scale=alt.Scale(domain=[-max_abs_delta, max_abs_delta]),
            ),
            y=alt.Y(
                "Label:N",
                title=None,
                axis=alt.Axis(labelLimit=360),
                sort=alt.EncodingSortField(field="Feature order", order="ascending"),
            ),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Increase", "Decrease"], range=["#2f7d5c", "#b34d4d"]),
                legend=None,
            ),
            tooltip=["Feature", "Compound", "Before", "After", "After - Before"],
        )
        .properties(height=ft_change_height)
    )
    zero_rule = alt.Chart(chart_data).mark_rule(color="#444444", strokeWidth=2).encode(x="Zero:Q")

    st.altair_chart(zero_rule + bars, use_container_width=True)
    table = ft_change_table
    if show_table_expander:
        with st.expander("Show FT change table", expanded=False):
            st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)


st.set_page_config(page_title="MG Gene NMF Prediction", page_icon="MG", layout="wide")

st.title("MG Gene NMF Single Case Prediction")
st.caption("Fill in one subject's before / after values, then run the notebook NMF and XGBoost prediction pipeline.")

core, reference, demo_cases = load_model_assets()
selected_features = list(core["Sel_Feature_list"])
asv_family_map, asv_family_order = load_asv_family_info()
case_options = [MANUAL_CASE_LABEL, *demo_cases.keys()]
default_case_label = "Positive - Demo_Pos_Case1" if "Positive - Demo_Pos_Case1" in case_options else case_options[0]


def case_for_label(case_label: str) -> dict[str, dict[str, float]]:
    if case_label == MANUAL_CASE_LABEL:
        return st.session_state.get("manual_case_data", empty_case(selected_features))
    return demo_cases[case_label]


def load_case_widget_values(case_label: str, case: dict[str, dict[str, float]]) -> None:
    for feature in selected_features:
        values = case.get(feature, {})
        st.session_state[input_widget_key(case_label, feature, "before")] = float(values.get("before", 0) or 0)
        st.session_state[input_widget_key(case_label, feature, "after")] = float(values.get("after", 0) or 0)


def clear_prediction_state() -> None:
    st.session_state.pop("last_prediction_result", None)
    st.session_state.pop("last_prediction_error", None)


def handle_case_selector_change() -> None:
    selected = st.session_state["case_selector"]
    if selected == MANUAL_CASE_LABEL:
        st.session_state.setdefault("manual_case_data", empty_case(selected_features))
    load_case_widget_values(selected, case_for_label(selected))
    clear_prediction_state()


def handle_input_value_change(source_case: str, feature: str, field: str) -> None:
    manual_case = {}
    baseline_case = case_for_label(source_case)

    for current_feature in selected_features:
        baseline_values = baseline_case.get(current_feature, {})
        before_key = input_widget_key(source_case, current_feature, "before")
        after_key = input_widget_key(source_case, current_feature, "after")
        manual_case[current_feature] = {
            "before": float(st.session_state.get(before_key, baseline_values.get("before", 0)) or 0),
            "after": float(st.session_state.get(after_key, baseline_values.get("after", 0)) or 0),
        }

    changed_key = input_widget_key(source_case, feature, field)
    manual_case[feature][field] = float(st.session_state.get(changed_key, 0) or 0)
    st.session_state["manual_case_data"] = manual_case
    st.session_state["case_selector"] = MANUAL_CASE_LABEL
    load_case_widget_values(MANUAL_CASE_LABEL, manual_case)
    clear_prediction_state()


if "manual_case_data" not in st.session_state:
    st.session_state["manual_case_data"] = empty_case(selected_features)

if "case_selector" not in st.session_state:
    st.session_state["case_selector"] = default_case_label
    load_case_widget_values(default_case_label, case_for_label(default_case_label))

if st.session_state.pop("force_manual_case_selector", False):
    st.session_state["case_selector"] = MANUAL_CASE_LABEL
    load_case_widget_values(MANUAL_CASE_LABEL, st.session_state["manual_case_data"])

with st.sidebar:
    st.header("Case setup")
    default_case_index = case_options.index(default_case_label)
    selected_case = st.selectbox(
        "Select a case",
        case_options,
        index=default_case_index,
        key="case_selector",
        on_change=handle_case_selector_change,
    )
    st.metric("Input features", len(selected_features))

initial_case = case_for_label(selected_case)

input_table = case_dict_to_table(initial_case, selected_features, reference, asv_family_map, asv_family_order)
edited_table = collect_sidebar_case(input_table, selected_case, on_value_change=handle_input_value_change)

if selected_case != MANUAL_CASE_LABEL and case_values_changed(edited_table, input_table):
    st.session_state["manual_case_data"] = table_to_case(edited_table)
    st.session_state["force_manual_case_selector"] = True
    clear_prediction_state()
    st.rerun()

non_zero_count = int(((edited_table["before"] != 0) | (edited_table["after"] != 0)).sum())
with st.sidebar:
    st.divider()
    st.info(f"{non_zero_count} features currently have non-zero before/after values.")
    predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)

prediction_result = None
prediction_error = None
if predict_clicked:
    if non_zero_count == 0:
        prediction_result = {
            "probability": None,
            "prediction": 0,
            "label": "Non-Improve (Negative)",
            "forced_no_input": True,
        }
        prediction_error = None
    else:
        try:
            prediction_result = predict_case(table_to_case(edited_table), core, reference)
            prediction_result["forced_no_input"] = False
            prediction_error = None
        except Exception as exc:
            prediction_result = None
            prediction_error = exc

    st.session_state["last_prediction_result"] = prediction_result
    st.session_state["last_prediction_error"] = prediction_error
elif "last_prediction_result" in st.session_state or "last_prediction_error" in st.session_state:
    prediction_result = st.session_state.get("last_prediction_result")
    prediction_error = st.session_state.get("last_prediction_error")

has_prediction_output = prediction_result is not None or prediction_error is not None

summary_tab, result_tab, probability_tab, change_tab, family_total_tab, family_clr_value_tab, family_clr_tab, input_tab = st.tabs(
    [
        "Summary",
        "Prediction result",
        "Probability",
        "Input change",
        "ASV family totals",
        "ASV family CLR values",
        "ASV family CLR change",
        "Current input",
    ]
)

with summary_tab:
    render_user_guide_download()

    if has_prediction_output:
        st.divider()
        st.subheader("Prediction result")
        render_prediction_summary(prediction_result, prediction_error)
        st.divider()
    else:
        st.caption("Charts update from the current input and do not require running prediction first.")

    asv_summary_tab, ft_summary_tab = st.tabs(["ASV Summary", "FT Summary"])

    with asv_summary_tab:
        render_family_clr_values_section(edited_table, float(core["ImputeValue"]), show_table_expander=True)
        st.divider()
        render_family_clr_change_section(edited_table, float(core["ImputeValue"]), show_table_expander=True)

    with ft_summary_tab:
        render_ft_values_section(edited_table, show_table_expander=True)
        st.divider()
        render_ft_change_section(edited_table, show_table_expander=True)

with input_tab:
    st.subheader("Current case input")
    st.dataframe(edited_table.drop(columns=["family_order"]), use_container_width=True, hide_index=True)

with change_tab:
    st.subheader("Before / After input change")
    include_zero_features = st.checkbox("Show zero-value features", value=False)
    change_table = build_input_change_table(edited_table, include_zero_features)

    if change_table.empty:
        st.info("No non-zero before/after values to plot yet.")
    else:
        chart_data = change_table.rename(
            columns={
                "label": "Feature",
                "type": "Type",
                "family": "Family",
                "before": "Before",
                "after": "After",
                "delta": "After - Before",
            }
        ).copy()
        chart_data["Direction"] = np.where(chart_data["After - Before"] >= 0, "Increase", "Decrease")
        chart_data["Sort value"] = chart_data["After - Before"].abs()
        chart_data["Zero"] = 0
        chart_height = max(320, min(900, len(chart_data) * 28))
        max_abs_delta = float(chart_data["After - Before"].abs().max() or 1)

        bars = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "After - Before:Q",
                    title="After - Before",
                    axis=alt.Axis(grid=True),
                    scale=alt.Scale(domain=[-max_abs_delta, max_abs_delta]),
                ),
                y=alt.Y(
                    "Feature:N",
                    title=None,
                    sort=alt.EncodingSortField(field="Sort value", order="descending"),
                ),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(domain=["Increase", "Decrease"], range=["#2f7d5c", "#b34d4d"]),
                    legend=None,
                ),
                tooltip=["Feature", "Type", "Family", "Before", "After", "After - Before"],
            )
            .properties(height=chart_height)
        )

        zero_rule = (
            alt.Chart(chart_data)
            .mark_rule(color="#444444", strokeWidth=2)
            .encode(x="Zero:Q")
        )

        change_chart = zero_rule + bars
        st.altair_chart(change_chart, use_container_width=True)

        st.dataframe(
            chart_data.drop(columns=["Direction", "Sort value", "Zero"]),
            use_container_width=True,
            hide_index=True,
        )

with family_total_tab:
    st.subheader("ASV family Before / After totals")
    st.caption("This chart updates from the current input and does not require running prediction first.")
    family_total_table = build_asv_family_total_table(edited_table)

    if family_total_table.empty:
        st.info("No ASV family values to plot yet.")
    else:
        family_total_long = family_total_table.melt(
            id_vars=["Family", "family_order"],
            value_vars=["Before total", "After total"],
            var_name="Timepoint",
            value_name="Total",
        )
        family_total_long["Timepoint"] = family_total_long["Timepoint"].str.replace(" total", "", regex=False)
        family_total_long["Timepoint order"] = family_total_long["Timepoint"].map({"Before": 0, "After": 1})
        family_total_height = max(420, min(1000, len(family_total_table) * 42))

        family_total_chart = (
            alt.Chart(family_total_long)
            .mark_bar()
            .encode(
                x=alt.X("Total:Q", title="Family total"),
                y=alt.Y(
                    "Family:N",
                    title=None,
                    axis=alt.Axis(labelLimit=320),
                    sort=alt.EncodingSortField(field="family_order", order="ascending"),
                ),
                yOffset=alt.YOffset(
                    "Timepoint:N",
                    sort=alt.EncodingSortField(field="Timepoint order", order="ascending"),
                ),
                color=alt.Color(
                    "Timepoint:N",
                    scale=alt.Scale(domain=["Before", "After"], range=["#2f6fbb", "#c94c4c"]),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["Family", "Timepoint", "Total"],
            )
            .properties(height=family_total_height)
        )

        st.altair_chart(family_total_chart, use_container_width=True)
        st.dataframe(
            family_total_table.drop(columns=["family_order"]),
            use_container_width=True,
            hide_index=True,
        )

with family_clr_value_tab:
    st.subheader("ASV family CLR Before / After values")
    st.caption("Family ASV values are summed first, then Before and After are CLR-transformed separately.")
    family_clr_value_table = build_asv_family_clr_value_table(
        edited_table,
        float(core["ImputeValue"]),
    )

    if family_clr_value_table.empty:
        st.info("No ASV family CLR values to plot yet.")
    else:
        family_clr_value_long = family_clr_value_table.melt(
            id_vars=["Family", "family_order"],
            value_vars=["Before CLR", "After CLR"],
            var_name="Timepoint",
            value_name="CLR value",
        )
        family_clr_value_long["Timepoint"] = family_clr_value_long["Timepoint"].str.replace(" CLR", "", regex=False)
        family_clr_value_long["Timepoint order"] = family_clr_value_long["Timepoint"].map({"Before": 0, "After": 1})
        clr_value_height = max(420, min(1000, len(family_clr_value_table) * 42))

        family_clr_value_chart = (
            alt.Chart(family_clr_value_long)
            .mark_bar()
            .encode(
                x=alt.X("CLR value:Q", title="CLR value"),
                y=alt.Y(
                    "Family:N",
                    title=None,
                    axis=alt.Axis(labelLimit=320),
                    sort=alt.EncodingSortField(field="family_order", order="ascending"),
                ),
                yOffset=alt.YOffset(
                    "Timepoint:N",
                    sort=alt.EncodingSortField(field="Timepoint order", order="ascending"),
                ),
                color=alt.Color(
                    "Timepoint:N",
                    scale=alt.Scale(domain=["Before", "After"], range=["#2f6fbb", "#c94c4c"]),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["Family", "Timepoint", "CLR value"],
            )
            .properties(height=clr_value_height)
        )

        st.altair_chart(family_clr_value_chart, use_container_width=True)
        st.dataframe(
            family_clr_value_table.drop(columns=["family_order"]),
            use_container_width=True,
            hide_index=True,
        )

with family_clr_tab:
    st.subheader("ASV family CLR change")
    family_clr_table = build_asv_family_clr_change_table(
        edited_table,
        float(core["ImputeValue"]),
        include_zero_families=True,
    )

    if family_clr_table.empty:
        st.info("No non-zero ASV family values to plot yet.")
    else:
        family_chart_data = family_clr_table.copy()
        family_chart_data["Direction"] = np.where(family_chart_data["CLR delta"] >= 0, "Increase", "Decrease")
        family_chart_data["Zero"] = 0
        family_chart_height = max(320, min(900, len(family_chart_data) * 30))
        max_abs_clr_delta = float(family_chart_data["CLR delta"].abs().max() or 1)

        family_bars = (
            alt.Chart(family_chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "CLR delta:Q",
                    title="CLR(After) - CLR(Before)",
                    axis=alt.Axis(grid=True),
                    scale=alt.Scale(domain=[-max_abs_clr_delta, max_abs_clr_delta]),
                ),
                y=alt.Y(
                    "Family:N",
                    title=None,
                    axis=alt.Axis(labelLimit=320),
                    sort=alt.EncodingSortField(field="family_order", order="ascending"),
                ),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(domain=["Increase", "Decrease"], range=["#2f7d5c", "#b34d4d"]),
                    legend=None,
                ),
                tooltip=["Family", "Before total", "After total", "Before CLR", "After CLR", "CLR delta"],
            )
            .properties(height=family_chart_height)
        )

        family_zero_rule = (
            alt.Chart(family_chart_data)
            .mark_rule(color="#444444", strokeWidth=2)
            .encode(x="Zero:Q")
        )

        st.altair_chart(family_zero_rule + family_bars, use_container_width=True)
        st.dataframe(
            family_chart_data.drop(columns=["Direction", "family_order", "Zero"]),
            use_container_width=True,
            hide_index=True,
        )

with result_tab:
    st.subheader("Prediction result")
    if not has_prediction_output:
        st.info("Fill the values in the sidebar, then press Run prediction.")

    if has_prediction_output:
        if prediction_error is not None:
            st.error("Prediction failed. Please check the input values.")
            st.exception(prediction_error)
        else:
            prediction = prediction_result["prediction"]

            st.metric("Prediction", prediction_result["label"])

            if prediction_result.get("forced_no_input"):
                st.warning("No input values were provided, so this case is forced to Non-Improve (Negative).")
            elif prediction == 1:
                st.success("This case is predicted as Improve (Positive).")
            else:
                st.warning("This case is predicted as Non-Improve (Negative).")

with probability_tab:
    st.subheader("Probability reference")
    if not has_prediction_output:
        st.info("Run prediction first to show the model probability.")

    if has_prediction_output:
        if prediction_error is not None:
            st.error("Prediction failed. Please check the input values.")
            st.exception(prediction_error)
        elif prediction_result.get("forced_no_input"):
            st.info("No model probability is shown because no input values were provided.")
        else:
            st.metric("Improve probability", f"{prediction_result['probability']:.6f}")
            st.caption("This value is shown separately as a reference for interpretation.")

            with st.expander("Show model intermediate outputs"):
                st.write("NMF selected components")
                st.dataframe(prediction_result["nmf_components"], use_container_width=True)
                st.write("Reconstructed features for XGBoost")
                st.dataframe(prediction_result["reconstructed_features"], use_container_width=True)
