from pathlib import Path

import pandas as pd
from django.contrib import messages
from django.shortcuts import redirect, render

from .forms import AddInteractionForm
from .models import InteractionLog
from .services import (
	NOTEBOOK6_RESULTS,
	load_data,
	mine_rules,
	recommend_user_cf,
	resolve_processed_dir,
	save_app_interaction,
)


def _missing_context() -> dict:
	return {
		"data_ready": False,
		"missing_msg": "Processed data not found. Run Notebook 1 to create data/processed files.",
	}


def dashboard(request):
	frames = load_data()
	if frames is None:
		return render(request, "management_portal/dashboard.html", _missing_context())

	user_data = frames["user_data"]
	product_data = frames["product_data"]
	matrix = frames["user_item_matrix_filled"]

	num_users = user_data["UserID"].nunique()
	num_products = product_data["ProductID"].nunique()
	interactions = len(user_data)
	possible = matrix.shape[0] * matrix.shape[1] if not matrix.empty else 0
	density = ((matrix > 0).sum().sum() / possible) if possible else 0.0

	top_categories = user_data["Category"].value_counts().head(8).reset_index()
	top_categories.columns = ["Category", "Interactions"]

	return render(
		request,
		"management_portal/dashboard.html",
		{
			"data_ready": True,
			"stats": {
				"users": num_users,
				"products": num_products,
				"interactions": interactions,
				"density": density,
			},
			"top_categories": top_categories.to_dict(orient="records"),
			"model_results": NOTEBOOK6_RESULTS,
		},
	)


def users_view(request):
	frames = load_data()
	if frames is None:
		return render(request, "management_portal/users.html", _missing_context())

	user_data = frames["user_data"].copy()
	product_data = frames["product_data"].copy()
	processed_dir = resolve_processed_dir()

	if request.method == "POST":
		form = AddInteractionForm(request.POST)
		if form.is_valid():
			user_id = form.cleaned_data["user_id"].strip()
			product_id = form.cleaned_data["product_id"].strip()
			rating = float(form.cleaned_data["rating"])

			if product_id not in set(product_data["ProductID"].astype(str)):
				messages.error(request, "ProductID not found in product catalog.")
				return redirect("management_portal:users")

			category_map = product_data.set_index("ProductID")["Category"].to_dict()
			new_row = pd.DataFrame(
				[
					{
						"UserID": user_id,
						"ProductID": product_id,
						"Category": category_map.get(product_id, "Unknown"),
						"Rating": rating,
						"Timestamp": pd.Timestamp.utcnow(),
					}
				]
			)
			updated = pd.concat([user_data, new_row], ignore_index=True)

			if processed_dir is not None:
				save_app_interaction(updated, Path(processed_dir))

			InteractionLog.objects.create(user_id=user_id, product_id=product_id, rating=rating)
			messages.success(request, "Interaction saved successfully.")
			return redirect("management_portal:users")
	else:
		form = AddInteractionForm()

	users_summary = (
		user_data.groupby("UserID")
		.size()
		.reset_index(name="Interactions")
		.sort_values("Interactions", ascending=False)
		.head(25)
	)
	products = product_data[["ProductID", "ProductName", "Category"]].head(300)
	recent_logs = InteractionLog.objects.all()[:20]

	return render(
		request,
		"management_portal/users.html",
		{
			"data_ready": True,
			"form": form,
			"users_summary": users_summary.to_dict(orient="records"),
			"products": products.to_dict(orient="records"),
			"recent_logs": recent_logs,
		},
	)


def recommendations_view(request):
	frames = load_data()
	if frames is None:
		return render(request, "management_portal/recommendations.html", _missing_context())

	product_data = frames["product_data"]
	matrix = frames["user_item_matrix_filled"]

	users = sorted(matrix.index.astype(str).tolist()) if not matrix.empty else []
	selected_user = request.GET.get("user_id", users[0] if users else "")

	try:
		k = max(1, min(20, int(request.GET.get("k", "5"))))
	except ValueError:
		k = 5

	recs = pd.DataFrame(columns=["ProductID", "PredictedScore"])
	if selected_user:
		recs = recommend_user_cf(selected_user, matrix, n=k)

	if not recs.empty:
		lookup = product_data.set_index("ProductID")[["ProductName", "Category"]]
		recs = recs.join(lookup, on="ProductID")
		recs["PredictedScore"] = recs["PredictedScore"].round(4)

	return render(
		request,
		"management_portal/recommendations.html",
		{
			"data_ready": True,
			"users": users,
			"selected_user": selected_user,
			"k": k,
			"recommendations": recs.to_dict(orient="records"),
		},
	)


def rules_view(request):
	frames = load_data()
	if frames is None:
		return render(request, "management_portal/rules.html", _missing_context())

	user_data = frames["user_data"]
	user_category_agg = frames["user_category_agg"]

	rules_all = mine_rules(user_data, min_support=0.08, min_conf=0.35)
	rules_show = rules_all.head(20).copy()
	if not rules_show.empty:
		rules_show["antecedents"] = rules_show["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
		rules_show["consequents"] = rules_show["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
		for c in ["support", "confidence", "lift"]:
			rules_show[c] = rules_show[c].astype(float).round(4)

	user_primary_group = (
		user_category_agg.sort_values(["UserID", "TotalInteractions"], ascending=[True, False])
		.drop_duplicates(subset="UserID")
		[["UserID", "Category"]]
		.rename(columns={"Category": "UserGroup"})
	)

	user_group_data = user_data[["UserID", "ProductID", "Timestamp"]].merge(user_primary_group, on="UserID", how="left")
	segment_rows = []
	for group_name, gdf in user_group_data.groupby("UserGroup"):
		g_rules = mine_rules(gdf[["UserID", "ProductID", "Timestamp"]], min_support=0.10, min_conf=0.35)
		top = g_rules.head(3).copy()
		if top.empty:
			continue
		top["UserGroup"] = group_name
		segment_rows.append(top)

	segment_rules = pd.concat(segment_rows, ignore_index=True) if segment_rows else pd.DataFrame()
	if not segment_rules.empty:
		segment_rules["antecedents"] = segment_rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
		segment_rules["consequents"] = segment_rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
		for c in ["support", "confidence", "lift"]:
			segment_rules[c] = segment_rules[c].astype(float).round(4)

	return render(
		request,
		"management_portal/rules.html",
		{
			"data_ready": True,
			"global_rules_count": len(rules_all),
			"rules": rules_show.to_dict(orient="records"),
			"segment_rules": segment_rules.head(15).to_dict(orient="records"),
		},
	)


def results_view(request):
	return render(
		request,
		"management_portal/results.html",
		{"model_results": NOTEBOOK6_RESULTS},
	)
