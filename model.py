import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

base = Path(".")
inj_path = base / "oklahoma_injection_joined_2011_2025.csv"
eq_path = base / "query (2).csv"
out_dir = base / "oklahoma_ml_13km_magnitude_models_improved"
out_dir.mkdir(exist_ok=True)

months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
inj_cols = ["API","LAT","LON","ReportYear","DataYear","InjTopDepth","InjBotDepth"] + [f"{m} Vol" for m in months] + [f"{m} PSI" for m in months]
eq_cols = ["time","latitude","longitude","mag"]

inj = pd.read_csv(inj_path, usecols=inj_cols)
eq = pd.read_csv(eq_path, usecols=eq_cols)

eq["time"] = pd.to_datetime(eq["time"], utc=True, errors="coerce").dt.tz_convert(None)
eq = eq.dropna(subset=["time","latitude","longitude","mag"]).copy()
eq = eq[(eq["latitude"].between(-90,90)) & (eq["longitude"].between(-180,180))].copy()
eq["month"] = eq["time"].dt.to_period("M").dt.to_timestamp()

inj = inj[(inj["LAT"].between(-90,90)) & (inj["LON"].between(-180,180))].copy()
inj["ReportYear"] = pd.to_datetime(inj["ReportYear"], errors="coerce")
inj["DataYear"] = pd.to_numeric(inj["DataYear"], errors="coerce")
inj["year"] = inj["ReportYear"].dt.year.fillna(inj["DataYear"]).astype("Int64")
inj["inj_depth_mid"] = inj[["InjTopDepth","InjBotDepth"]].mean(axis=1)

parts = []
for i, m in enumerate(months, start=1):
    p = inj[["API","LAT","LON","year","inj_depth_mid",f"{m} Vol",f"{m} PSI"]].copy()
    p.columns = ["API","LAT","LON","year","inj_depth_mid","volume","pressure"]
    p["month_num"] = i
    parts.append(p)

well_monthly = pd.concat(parts, ignore_index=True)
well_monthly = well_monthly.dropna(subset=["year"]).copy()
well_monthly["month"] = pd.to_datetime(
    dict(year=well_monthly["year"].astype(int), month=well_monthly["month_num"], day=1),
    errors="coerce"
)
well_monthly = well_monthly.dropna(subset=["month","API","LAT","LON","inj_depth_mid"]).copy()
well_monthly["volume"] = pd.to_numeric(well_monthly["volume"], errors="coerce")
well_monthly["pressure"] = pd.to_numeric(well_monthly["pressure"], errors="coerce")
well_monthly = well_monthly.dropna(subset=["volume","pressure"]).copy()
well_monthly = well_monthly[(well_monthly["volume"] >= 0) & (well_monthly["pressure"] >= 0)].copy()
well_monthly = well_monthly[(well_monthly["month"] >= eq["month"].min()) & (well_monthly["month"] <= eq["month"].max())].copy()

wells_static = well_monthly.groupby("API", as_index=False).agg(
    LAT=("LAT","first"),
    LON=("LON","first")
)

tree = BallTree(np.deg2rad(eq[["latitude","longitude"]].to_numpy()), metric="haversine")
indices = tree.query_radius(
    np.deg2rad(wells_static[["LAT","LON"]].to_numpy()),
    r=13 / 6371.0088,
    return_distance=False
)

lengths = np.fromiter((len(x) for x in indices), dtype=int, count=len(indices))
well_idx = np.repeat(np.arange(len(indices)), lengths)
quake_idx = np.concatenate(indices) if lengths.sum() else np.array([], dtype=int)

pairs = pd.DataFrame({
    "API": wells_static["API"].to_numpy()[well_idx],
    "month": eq["month"].to_numpy()[quake_idx],
    "mag": eq["mag"].to_numpy()[quake_idx],
})

mag_monthly = pairs.groupby(["API","month"], as_index=False).agg(
    avg_mag_13km=("mag","mean"),
    quake_count_13km=("mag","size"),
)

df = well_monthly.merge(mag_monthly, on=["API","month"], how="left")
df["avg_mag_13km"] = df["avg_mag_13km"].fillna(0.0)
df["quake_count_13km"] = df["quake_count_13km"].fillna(0.0)
df = df.sort_values(["API","month"]).copy()

for lag in [1,2,3]:
    df[f"lag{lag}_avg_mag_13km"] = df.groupby("API")["avg_mag_13km"].shift(lag)
    df[f"lag{lag}_count_13km"] = df.groupby("API")["quake_count_13km"].shift(lag)

df["lag1_volume"] = df.groupby("API")["volume"].shift(1)
df["lag1_pressure"] = df.groupby("API")["pressure"].shift(1)
df["roll3_avg_mag_13km"] = (
    df.groupby("API")["avg_mag_13km"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
)
df["log_volume"] = np.log1p(df["volume"])
df["depth_x_volume"] = df["inj_depth_mid"] * df["log_volume"]
df["depth_x_pressure"] = df["inj_depth_mid"] * df["pressure"]
df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
df = df.fillna(0.0)

sample = df.sample(min(50000, len(df)), random_state=42).copy()

features = [
    "inj_depth_mid","volume","pressure","LAT","LON",
    "lag1_avg_mag_13km","lag2_avg_mag_13km","lag3_avg_mag_13km",
    "lag1_count_13km","lag2_count_13km","lag3_count_13km",
    "lag1_volume","lag1_pressure","roll3_avg_mag_13km",
    "log_volume","depth_x_volume","depth_x_pressure","month_sin","month_cos"
]

X = sample[features]
y = sample["avg_mag_13km"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = {
    "Dummy mean": DummyRegressor(strategy="mean"),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Extra trees improved": ExtraTreesRegressor(
        n_estimators=80,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    ),
}

rows = []
preds = {}
trained = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "model": name,
        "r2": r2_score(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "mae": mean_absolute_error(y_test, pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    })
    preds[name] = pred
    trained[name] = model

metrics_df = pd.DataFrame(rows).sort_values(["r2","rmse"], ascending=[False, True])
metrics_df.to_csv(out_dir / "improved_magnitude_model_comparison_metrics.csv", index=False)
sample.to_csv(out_dir / "improved_magnitude_modeling_table_sample.csv", index=False)

best_model_name = metrics_df.iloc[0]["model"]
best_model = trained[best_model_name]
best_pred = preds[best_model_name]

plt.figure(figsize=(8,5))
plt.bar(metrics_df["model"], metrics_df["r2"])
plt.ylabel("R²")
plt.title("Improved 13 km average-magnitude model comparison")
plt.tight_layout()
plt.savefig(out_dir / "improved_magnitude_model_comparison_r2.png", dpi=180)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(metrics_df["model"], metrics_df["rmse"])
plt.ylabel("RMSE")
plt.title("Improved 13 km average-magnitude model comparison")
plt.tight_layout()
plt.savefig(out_dir / "improved_magnitude_model_comparison_rmse.png", dpi=180)
plt.close()

sel = np.random.RandomState(42).choice(len(y_test), size=min(4000, len(y_test)), replace=False)
yy = np.asarray(y_test)[sel]
pp = np.asarray(best_pred)[sel]

plt.figure(figsize=(7,6))
plt.scatter(yy, pp, alpha=0.25)
mn = min(yy.min(), pp.min())
mx = max(yy.max(), pp.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual average magnitude")
plt.ylabel("Predicted average magnitude")
plt.title(f"Actual vs predicted: {best_model_name}")
plt.tight_layout()
plt.savefig(out_dir / "improved_best_model_actual_vs_pred.png", dpi=180)
plt.close()

resid = np.asarray(y_test) - np.asarray(best_pred)
plt.figure(figsize=(8,5))
plt.hist(resid, bins=30)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title(f"Residuals: {best_model_name}")
plt.tight_layout()
plt.savefig(out_dir / "improved_best_model_residuals.png", dpi=180)
plt.close()

if "Extra trees" in best_model_name:
    imp = best_model.feature_importances_
else:
    coef = np.abs(best_model.named_steps["model"].coef_)
    imp = coef / coef.sum() if coef.sum() else np.zeros_like(coef)

imp_df = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance", ascending=False)
imp_df.to_csv(out_dir / "improved_best_model_feature_importance.csv", index=False)

plt.figure(figsize=(9,5))
top = imp_df.head(10)
plt.bar(top["feature"], top["importance"])
plt.xticks(rotation=25, ha="right")
plt.ylabel("Importance")
plt.title(f"Top feature importance: {best_model_name}")
plt.tight_layout()
plt.savefig(out_dir / "improved_best_model_feature_importance.png", dpi=180)
plt.close()

lat_min, lat_max = sample["LAT"].quantile(0.02), sample["LAT"].quantile(0.98)
lon_min, lon_max = sample["LON"].quantile(0.02), sample["LON"].quantile(0.98)
lat_grid = np.linspace(lat_min, lat_max, 35)
lon_grid = np.linspace(lon_min, lon_max, 45)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

grid = pd.DataFrame({
    "inj_depth_mid": np.full(lat_mesh.size, sample["inj_depth_mid"].median()),
    "volume": np.full(lat_mesh.size, sample["volume"].median()),
    "pressure": np.full(lat_mesh.size, sample["pressure"].median()),
    "LAT": lat_mesh.ravel(),
    "LON": lon_mesh.ravel(),
    "lag1_avg_mag_13km": np.full(lat_mesh.size, sample["lag1_avg_mag_13km"].median()),
    "lag2_avg_mag_13km": np.full(lat_mesh.size, sample["lag2_avg_mag_13km"].median()),
    "lag3_avg_mag_13km": np.full(lat_mesh.size, sample["lag3_avg_mag_13km"].median()),
    "lag1_count_13km": np.full(lat_mesh.size, sample["lag1_count_13km"].median()),
    "lag2_count_13km": np.full(lat_mesh.size, sample["lag2_count_13km"].median()),
    "lag3_count_13km": np.full(lat_mesh.size, sample["lag3_count_13km"].median()),
    "lag1_volume": np.full(lat_mesh.size, sample["lag1_volume"].median()),
    "lag1_pressure": np.full(lat_mesh.size, sample["lag1_pressure"].median()),
    "roll3_avg_mag_13km": np.full(lat_mesh.size, sample["roll3_avg_mag_13km"].median()),
    "log_volume": np.full(lat_mesh.size, sample["log_volume"].median()),
    "depth_x_volume": np.full(lat_mesh.size, sample["depth_x_volume"].median()),
    "depth_x_pressure": np.full(lat_mesh.size, sample["depth_x_pressure"].median()),
    "month_sin": np.full(lat_mesh.size, sample["month_sin"].median()),
    "month_cos": np.full(lat_mesh.size, sample["month_cos"].median()),
})
grid_pred = best_model.predict(grid[features]).reshape(lat_mesh.shape)

plt.figure(figsize=(10,6))
plt.contourf(lon_mesh, lat_mesh, grid_pred, levels=12)
plt.colorbar(label="Predicted average magnitude in 13 km")
pts = sample[["LON","LAT"]].drop_duplicates().sample(min(1000, len(sample[["LON","LAT"]].drop_duplicates())), random_state=42)
plt.scatter(pts["LON"], pts["LAT"], s=3, alpha=0.15)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Improved Oklahoma heat map ({best_model_name})")
plt.tight_layout()
plt.savefig(out_dir / "improved_oklahoma_magnitude_prediction_heatmap.png", dpi=180)
plt.close()

print(metrics_df)
print("Saved to", out_dir)