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
ASV_FAMILY_FILE = APP_DIR / "ASV_Family_name.csv"
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
FT_HMDB_MAP = {
    "FT0047": "HMDB0033951",
    "FT0270": "HMDB0000900",
    "FT0131": "HMDB0001476",
    "FT2032": "HMDB0000012",
    "FT1558": "HMDB0030524",
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
        **{f"Demo-Improver-Case {index}": case for index, case in enumerate(demo_positive.values(), start=1)},
        **{f"Demo-Non-Improver-Case {index}": case for index, case in enumerate(demo_negative.values(), start=1)},
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
    asv_sequence = 0

    for feature in selected_features:
        values = case.get(feature, {})
        feature_type = "ASV" if feature in asv_features else "FT"
        if feature_type == "ASV":
            asv_sequence += 1
        family = asv_family_map.get(feature, "Unmapped family") if feature_type == "ASV" else "FT features"
        rows.append(
            {
                "feature": feature,
                "type": feature_type,
                "family": family,
                "asv_sequence": asv_sequence if feature_type == "ASV" else None,
                "family_order": asv_family_order.get(family, 9999) if feature_type == "ASV" else 10000,
                "display_name": FT_NAME_MAP.get(feature, "") if feature_type == "FT" else "",
                "hmdb_id": FT_HMDB_MAP.get(feature, "") if feature_type == "FT" else "",
                "before": float(values.get("before", 0) or 0),
                "after": float(values.get("after", 0) or 0),
            }
        )

    return pd.DataFrame(rows)


def sidebar_feature_caption(row: Any) -> str:
    if row.type == "FT" and row.display_name:
        feature_label = row.hmdb_id if row.hmdb_id else row.feature
        return f"{feature_label} - {row.display_name}"
    return str(row.feature)


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
        display_family = format_family_with_count_markdown(str(family), len(family_table))
        groups[display_family] = family_table

    ft_table = input_table[input_table["type"] != "ASV"]
    if not ft_table.empty:
        groups["FT features"] = ft_table

    return groups


def collect_sidebar_case(input_table: pd.DataFrame, selected_case: str, on_value_change: Any | None = None) -> pd.DataFrame:
    edited_rows = []
    feature_groups = build_feature_groups(input_table)

    for group_name, group_table in feature_groups.items():
        is_asv_group = bool(group_table["type"].eq("ASV").all())
        non_zero_count = int(((group_table["before"] != 0) | (group_table["after"] != 0)).sum())
        if is_asv_group:
            label = group_name
        else:
            label = f"{group_name} ({len(group_table)} features"
        if non_zero_count and not is_asv_group:
            label += f", {non_zero_count} filled"
        if not is_asv_group:
            label += ")"

        with st.sidebar.expander(label, expanded=False):
            for row in group_table.itertuples(index=False):
                st.caption(sidebar_feature_caption(row))
                before_col, after_col = st.columns(2)
                before_label = "Before (E-08)" if row.type == "FT" else "Before"
                after_label = "After (E-08)" if row.type == "FT" else "After"
                before_value = before_col.number_input(
                    before_label,
                    min_value=0.0,
                    value=float(row.before),
                    format="%.8f",
                    key=input_widget_key(selected_case, row.feature, "before"),
                    on_change=on_value_change,
                    args=(selected_case, row.feature, "before") if on_value_change else None,
                )
                after_value = after_col.number_input(
                    after_label,
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
                        "asv_sequence": row.asv_sequence,
                        "family_order": row.family_order,
                        "display_name": row.display_name,
                        "hmdb_id": row.hmdb_id,
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
    change_table.loc[ft_mask, "feature_label"] = change_table.loc[ft_mask, "hmdb_id"].where(
        change_table.loc[ft_mask, "hmdb_id"] != "",
        change_table.loc[ft_mask, "feature"],
    )
    change_table.loc[ft_mask, "label"] = (
        change_table.loc[ft_mask, "feature_label"] + " - " + change_table.loc[ft_mask, "display_name"]
    )
    change_table["delta"] = change_table["after"] - change_table["before"]

    if not include_zero_features:
        change_table = change_table[(change_table["before"] != 0) | (change_table["after"] != 0)]

    return change_table[["label", "type", "family", "before", "after", "delta"]]


def build_ft_total_table(input_table: pd.DataFrame) -> pd.DataFrame:
    ft_table = input_table[input_table["type"] == "FT"].copy()
    if ft_table.empty:
        return pd.DataFrame(columns=["Feature", "Compound", "Before", "After"])

    ft_table["feature_label"] = np.where(ft_table["hmdb_id"] != "", ft_table["hmdb_id"], ft_table["feature"])
    ft_table["Compound"] = np.where(ft_table["display_name"] != "", ft_table["display_name"], ft_table["feature"])
    return ft_table.rename(columns={"feature_label": "Feature", "before": "Before", "after": "After"})[
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


def format_family_with_count_markdown(family: str, count: int | float) -> str:
    return f"*{clean_family_name(family)}* (N={int(count)})"


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

    prediction_col, probability_col, threshold_col = st.columns(3)
    prediction_col.metric("Prediction", prediction_result["label"])

    probability = prediction_result.get("probability")
    if probability is None:
        probability_col.metric("Improve probability", "N/A")
    else:
        probability_col.metric("Improve probability", f"{float(probability):.4%}")

    threshold = prediction_result.get("threshold")
    if threshold is None:
        threshold_col.metric("Threshold", "N/A")
    else:
        threshold_col.metric("Threshold", f"{float(threshold):.4%}")

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
    st.subheader("Family level")
    st.caption("Sum of family-level relative abundance changes")
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
    st.subheader("Difference")
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
        with st.expander("Show Difference Table", expanded=False):
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
    st.subheader("FT Difference")
    ft_change_table = build_ft_change_table(edited_input)

    if ft_change_table.empty:
        st.info("No FT differences to plot yet.")
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
        with st.expander("Show FT Difference table", expanded=False):
            st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.dataframe(table, use_container_width=True, hide_index=True)


def apply_readability_styles() -> None:
    st.markdown(
        """
        <style>
            html, body, .stApp {
                font-size: 19px !important;
            }

            .stApp p,
            .stApp span,
            .stApp label,
            .stApp button,
            .stApp input,
            .stApp textarea,
            .stApp [role="button"],
            .stApp [role="tab"],
            .stApp [role="combobox"],
            .stApp [data-baseweb="select"] *,
            .stApp [data-baseweb="input"] *,
            .stApp [data-testid="stMarkdownContainer"] *,
            .stApp [data-testid="stCaptionContainer"],
            .stApp [data-testid="stExpander"] *,
            .stApp [data-testid="stNotificationContent"] * {
                font-size: 1.08rem !important;
                line-height: 1.45 !important;
            }

            [data-testid="stSidebar"] {
                font-size: 1.08rem !important;
            }

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] button,
            [data-testid="stSidebar"] [role="combobox"] {
                font-size: 1.06rem !important;
                line-height: 1.4 !important;
            }

            h1 {
                font-size: 2.75rem !important;
            }

            h2 {
                font-size: 2rem !important;
            }

            h3 {
                font-size: 1.55rem !important;
            }

            [data-testid="stCaptionContainer"] {
                font-size: 1.05rem !important;
            }

            [data-testid="stMetricValue"] {
                font-size: 2.25rem !important;
            }

            .sidebar-app-title {
                font-size: 1.18rem !important;
                font-weight: 750;
                line-height: 1.24 !important;
                margin: 0.1rem 0 0.75rem;
            }

            .sidebar-author {
                font-size: 0.86rem !important;
                line-height: 1.35 !important;
                margin-bottom: 1.15rem;
            }

            .sidebar-author-label {
                font-weight: 750;
            }

            .sidebar-select-label {
                color: #1f2937 !important;
                font-size: 0.9rem !important;
                font-weight: 750;
                letter-spacing: 0.04em;
                margin-bottom: 0.2rem;
            }

            .sidebar-input-note {
                font-size: 0.86rem !important;
                line-height: 1.38 !important;
                margin: 0.65rem 0 0.9rem;
            }

            [data-testid="stDataFrame"],
            [data-testid="stDataFrame"] *,
            .stDataFrame,
            .stDataFrame * {
                font-size: 1.05rem !important;
            }

            .vega-embed text {
                font-size: 14px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="MG Gene NMF Prediction", page_icon="MG", layout="wide")
apply_readability_styles()

st.title("Analysis the association based on NLFXR")
st.caption("Fill in one subject's before / after values, then run the notebook NMF and XGBoost prediction pipeline.")

core, reference, demo_cases = load_model_assets()
selected_features = list(core["Sel_Feature_list"])
asv_family_map, asv_family_order = load_asv_family_info()
case_options = [MANUAL_CASE_LABEL, *demo_cases.keys()]
default_case_label = "Demo-Improver-Case 1" if "Demo-Improver-Case 1" in case_options else case_options[0]


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
    st.session_state["active_case_label"] = MANUAL_CASE_LABEL
    load_case_widget_values(MANUAL_CASE_LABEL, manual_case)
    clear_prediction_state()


if "manual_case_data" not in st.session_state:
    st.session_state["manual_case_data"] = empty_case(selected_features)

if "active_case_label" not in st.session_state:
    st.session_state["active_case_label"] = default_case_label
    load_case_widget_values(default_case_label, case_for_label(default_case_label))

if st.session_state.pop("force_manual_case_selector", False):
    st.session_state["active_case_label"] = MANUAL_CASE_LABEL
    load_case_widget_values(MANUAL_CASE_LABEL, st.session_state["manual_case_data"])

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-app-title">
            Longitudinal Gut Microbiome<br>
            Metabolome Explorer for Myasthenia Gravis<br>
            (GutLogMG)
        </div>
        <div class="sidebar-author">
            <span class="sidebar-author-label">Author:</span>
            <span class="sidebar-author-names">
                Che-Cheng Chang, Kuan-Yu Lin, Chien Ju Lin,
                Jiann-Horng Yeh, Hou-Chang Chiu, Tzu Chi Liu and Chi-Jie Lu
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-select-label">SELECT</div>', unsafe_allow_html=True)
    active_case_label = st.session_state.get("active_case_label", default_case_label)
    if active_case_label not in case_options:
        active_case_label = default_case_label
        st.session_state["active_case_label"] = active_case_label
    default_case_index = case_options.index(active_case_label)
    selected_case = st.selectbox(
        "SELECT",
        case_options,
        index=default_case_index,
        label_visibility="collapsed",
    )
    if selected_case != active_case_label:
        st.session_state["active_case_label"] = selected_case
        if selected_case == MANUAL_CASE_LABEL:
            st.session_state.setdefault("manual_case_data", empty_case(selected_features))
        load_case_widget_values(selected_case, case_for_label(selected_case))
        clear_prediction_state()
    st.markdown(
        """
        <div class="sidebar-input-note">
            Please input your data as follows:<br>
            (1) Sample sequence numbers (unit: number) in the ASV table corresponding to the ASV-ID<br>
            (2) Peak values (unit: E-08) in the MSMSID table corresponding to HMDB-ID
        </div>
        """,
        unsafe_allow_html=True,
    )

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

render_user_guide_download()

if has_prediction_output:
    st.divider()
    st.subheader("Prediction result")
    render_prediction_summary(prediction_result, prediction_error)
    st.divider()
else:
    st.caption("Charts update from the current input and do not require running prediction first.")

asv_summary_tab, ft_summary_tab = st.tabs(["ASV Summary", "Metabolites Summary"])

with asv_summary_tab:
    render_family_clr_values_section(edited_table, float(core["ImputeValue"]), show_table_expander=True)
    st.divider()
    render_family_clr_change_section(edited_table, float(core["ImputeValue"]), show_table_expander=True)

with ft_summary_tab:
    render_ft_values_section(edited_table, show_table_expander=True)
    st.divider()
    render_ft_change_section(edited_table, show_table_expander=True)
