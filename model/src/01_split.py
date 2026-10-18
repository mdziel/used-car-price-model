"""Phase 1: lock the validation design (hold-out sellers and repeated grouped folds). Run once; the output is frozen."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

df, names = lib.load()
d = lib.make_design(df)
d.to_csv(lib.RES + "folds.csv", index=False)
dev = d["holdout"] == 0
print("rows", len(d), "| hold-out rows", int((~dev).sum()), f"({(~dev).mean():.1%})", "| hold-out sellers", d.loc[~dev, "seller_id"].nunique())
for r in range(lib.N_REPEATS):
    f = d.loc[dev, f"fold_r{r}"]
    print(f"repeat {r}: fold sizes", f.value_counts().sort_index().tolist())
# check: no seller appears in two folds within a repeat, and none in both hold-out and development
for r in range(lib.N_REPEATS):
    g = d[dev].groupby("seller_id")[f"fold_r{r}"].nunique()
    assert (g == 1).all()
assert set(d.loc[dev, "seller_id"]).isdisjoint(set(d.loc[~dev, "seller_id"]))
h = df[~dev.values]
print("hold-out profile:", h["fuel_grp"].value_counts(normalize=True).round(2).to_dict(), "| dev:", df[dev.values]["fuel_grp"].value_counts(normalize=True).round(2).to_dict())
print("median price hold-out vs dev:", int(h["price"].median()), int(df[dev.values]["price"].median()))
print("chains:", df["chain"].replace("", "none").value_counts().head(12).to_dict())
print("sellers by chain_or_seller:", df["chain_or_seller"].nunique())
