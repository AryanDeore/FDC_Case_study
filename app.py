import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Load artifacts ---
qr_lower = joblib.load('model_artifacts/qr_lower.pkl')
qr_median = joblib.load('model_artifacts/qr_median.pkl')
qr_upper = joblib.load('model_artifacts/qr_upper.pkl')
scaler = joblib.load('model_artifacts/scaler.pkl')
te_company = joblib.load('model_artifacts/te_company.pkl')
te_sub_area = joblib.load('model_artifacts/te_sub_area.pkl')
tfidf = joblib.load('model_artifacts/tfidf.pkl')
expensive_centroid = joblib.load('model_artifacts/expensive_centroid.pkl')
cheap_centroid = joblib.load('model_artifacts/cheap_centroid.pkl')
q_hat = joblib.load('model_artifacts/q_hat.pkl')
model_feature_columns = joblib.load('model_artifacts/model_feature_columns.pkl')

# --- Dropdown options ---
COMPANIES = [
    'ace constructions', 'anp corp', 'bhaktamar realities', 'calyx spaces',
    'dolphin group', 'godrej properties', 'jhala group', 'kalpataru',
    'keystone landmark', 'kohinoor group', 'kundan spaces', 'lush life',
    'maha anand pinnac associates', 'majestique landmarks', 'mantra properties',
    'nirman developers', 'oxy buildcorp', 'porwal  develkoper',
    'porwal & anand develkoper', 'proviso group', 'puraniks', 'ravima ventures',
    'sagitarius ecospaces llp', 'shapoorji paloonji', 'shroff developers',
    'sukwani associates', 'supreme', 'tejraaj group', 'unique properties',
    'urban space creator', 'vasupujya corporation', 'venkatesh bhoomi construction',
    'vijaya laxmi creations', 'vijaya laxmi infrarealtors', 'vtp reality',
    'waghvani constructions', 'wellwisher apartments'
]

SUB_AREAS = [
    'akurdi', 'balewadi', 'baner', 'bavdhan', 'bavdhan budruk', 'bt kawade rd',
    'dhanori', 'hadapsar', 'handewadi', 'hinjewadi', 'karvanagar', 'kayani nagar',
    'keshav nagar', 'kharadi', 'kirkatwadi sinhagad road', 'kiwale',
    'koregaon park', 'koregoan', 'lonavala', 'magarpatta', 'mahalunge', 'manjri',
    'mohammadwadi', 'mundhwa', 'nibm', 'pisoli', 'ravet', 'susgaon', 'talegoan',
    'tathawade', 'undri', 'vimannagar', 'wadgaon sheri'
]


def normalize_text_value(value):
    """Mirror the notebook's text normalization for inference-time inputs."""
    return ' '.join(str(value or '').lower().strip().split())


def compute_desc_price_signal(description):
    """Compute description price signal using TF-IDF similarity."""
    description = normalize_text_value(description)
    if description == '':
        return 0.0
    tfidf_vec = tfidf.transform([description])
    sim_expensive = cosine_similarity(tfidf_vec, expensive_centroid.reshape(1, -1))[0][0]
    sim_cheap = cosine_similarity(tfidf_vec, cheap_centroid.reshape(1, -1))[0][0]
    return sim_expensive - sim_cheap


def predict(area_sqft, bedrooms, swimming_pool, mall, hospital, school,
            clubhouse, park_jogging_track, gym, company_name, sub_area,
            description, listing_price=None):
    """Run feature pipeline + inference, return price range and verdict."""

    # Binary amenities
    pool = 1 if swimming_pool else 0
    neighbourhood_amenities = sum([
        1 if mall else 0,
        1 if hospital else 0,
        1 if school else 0,
    ])
    property_amenities = sum([
        1 if clubhouse else 0,
        1 if park_jogging_track else 0,
        1 if gym else 0,
    ])

    # Description signal
    desc_signal = compute_desc_price_signal(description)

    # Log transform area
    area_log = np.log1p(area_sqft)

    company_name = normalize_text_value(company_name)
    sub_area = normalize_text_value(sub_area)

    # Target encode company and sub_area
    company_encoded = te_company.transform(
        pd.DataFrame({'company_name': [company_name]})
    )[0][0]
    sub_area_encoded = te_sub_area.transform(
        pd.DataFrame({'sub_area': [sub_area]})
    )[0][0]

    # Build feature vector using the saved training schema from the notebook.
    features = pd.DataFrame([{
        'swimming_pool': pool,
        'no_of_bedrooms': bedrooms,
        'neighbourhood_amenities': neighbourhood_amenities,
        'property_amenities': property_amenities,
        'desc_price_signal': desc_signal,
        'property_area_log': area_log,
        'te_company_name': company_encoded,
        'te_sub_area': sub_area_encoded,
    }]).reindex(columns=model_feature_columns, fill_value=0)

    # Scale (keep as DataFrame to preserve feature names)
    features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

    # Inference
    pred_lower = np.expm1(qr_lower.predict(features_scaled)[0] - q_hat)
    pred_median = np.expm1(qr_median.predict(features_scaled)[0])
    pred_upper = np.expm1(qr_upper.predict(features_scaled)[0] + q_hat)

    # Build result HTML
    result = (
        f"<div class='result-range'>{pred_lower:.1f} &ndash; {pred_upper:.1f} lakhs</div>"
        f"<div class='result-median'>Predicted price: {pred_median:.1f} lakhs</div>"
    )

    # Range bar (always shown)
    bar_min = pred_lower * 0.85
    bar_max = pred_upper * 1.15
    bar_span = bar_max - bar_min
    lower_pct = ((pred_lower - bar_min) / bar_span) * 100
    upper_pct = ((pred_upper - bar_min) / bar_span) * 100
    median_pct = ((pred_median - bar_min) / bar_span) * 100

    range_bar = (
        f"<div class='range-bar-container'>"
        f"<div class='range-bar-track'>"
        f"<div class='range-bar-fill' style='left: {lower_pct:.1f}%; width: {upper_pct - lower_pct:.1f}%;'></div>"
        f"<div class='range-bar-median' style='left: {median_pct:.1f}%;'></div>"
    )

    if listing_price is not None and listing_price > 0:
        marker_pct = max(0, min(100, ((listing_price - bar_min) / bar_span) * 100))
        range_bar += f"<div class='range-bar-marker' style='left: {marker_pct:.1f}%;'></div>"

    range_bar += (
        f"</div>"
        f"<div class='range-bar-labels'>"
        f"<span>{pred_lower:.1f}</span>"
        f"<span style='position: absolute; left: {median_pct:.1f}%; transform: translateX(-50%);'>{pred_median:.1f}</span>"
        f"<span>{pred_upper:.1f}</span>"
        f"</div>"
        f"</div>"
    )
    result += range_bar

    if listing_price is not None and listing_price > 0:
        range_width = pred_upper - pred_lower
        if listing_price < pred_lower:
            result += f"<div class='verdict verdict-under'>Below Range &mdash; {listing_price:.1f} lakhs is below the estimated range</div>"
        elif listing_price > pred_upper:
            result += f"<div class='verdict verdict-over'>Above Range &mdash; {listing_price:.1f} lakhs is above the estimated range</div>"
        elif listing_price < pred_lower + range_width * 0.25:
            result += f"<div class='verdict verdict-low'>Near Lower End &mdash; {listing_price:.1f} lakhs, in the lower quarter of the range</div>"
        elif listing_price > pred_upper - range_width * 0.25:
            result += f"<div class='verdict verdict-high'>Near Upper End &mdash; {listing_price:.1f} lakhs, in the upper quarter of the range</div>"
        else:
            result += f"<div class='verdict verdict-ok'>Within Range &mdash; {listing_price:.1f} lakhs, well within the estimated range</div>"
        result += "<div class='result-basis'>Your price is compared against the model's estimated range.</div>"
    else:
        result += "<div class='result-basis'>Based on area, bedrooms, builder, locality, amenities, and description.</div>"

    return result


# --- Feature-name labels for the waterfall chart ---
FEATURE_LABELS = {
    'swimming_pool': 'Swimming Pool',
    'no_of_bedrooms': 'Bedrooms',
    'neighbourhood_amenities': 'Neighbourhood Amenities',
    'property_amenities': 'Property Amenities',
    'desc_price_signal': 'Description Quality',
    'property_area_log': 'Property Area',
    'te_company_name': 'Builder',
    'te_sub_area': 'Location',
}

import re as _re
import html as _html


def _yn(val):
    """Convert messy yes/no column to bool."""
    return str(val).strip().lower().startswith('y')


def _extract_bedrooms(val):
    """Extract bedroom count from property type string."""
    s = _re.sub(r'[a-zA-Z\s]', '', str(val).strip())
    if '+' in s:
        parts = s.split('+')
        return sum(float(p) for p in parts if p)
    try:
        return float(s)
    except ValueError:
        return 0


def _get_contributions(features_scaled, pred_median):
    """Compute per-feature contributions in lakhs that sum exactly to pred_median.

    In log-space the prediction is: log_pred = intercept + sum(w_i * x_i).
    Each feature's share of the predicted price is:
        share_i = pred_median * (w_i * x_i) / log_pred
    The intercept's share is distributed proportionally across features
    (weighted by |share_i|) so contributions sum to pred_median with no
    leftover baseline.
    """
    coef = qr_median.coef_
    intercept = qr_median.intercept_
    scaled_vals = features_scaled.values[0]

    log_pred = intercept + sum(coef[i] * scaled_vals[i]
                               for i in range(len(coef)))
    if abs(log_pred) < 1e-9:
        return []

    # Each feature's proportional share of the predicted price
    raw = []
    for i, col in enumerate(model_feature_columns):
        share = pred_median * (coef[i] * scaled_vals[i]) / log_pred
        raw.append((FEATURE_LABELS.get(col, col), share))

    # Intercept's share — distribute among features by |share|
    intercept_share = pred_median * (intercept / log_pred)
    abs_total = sum(abs(s) for _, s in raw)
    if abs_total < 1e-9:
        return []

    contributions = []
    for label, share in raw:
        adjusted = share + intercept_share * (abs(share) / abs_total)
        if abs(adjusted) >= 0.1:
            contributions.append((label, adjusted))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


def _build_gap_card(listing_price, pred_median, pred_lower, pred_upper):
    """D: Highlighted gap card showing listed vs estimated."""
    gap = listing_price - pred_median
    abs_gap = abs(gap)

    if listing_price > pred_upper:
        gap_class = 'gap-over'
        gap_icon = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
        gap_msg = f"Nothing in this property's features justifies a {abs_gap:.0f}L premium over comparable listings."
    elif listing_price < pred_lower:
        gap_class = 'gap-under'
        gap_icon = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>'
        gap_msg = f"Priced {abs_gap:.0f}L below the estimated value &mdash; worth investigating why."
    else:
        gap_class = 'gap-fair'
        gap_icon = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>'
        gap_msg = "This listing is priced within the expected range for its features."

    sign = "+" if gap > 0 else ""

    return (
        f"<div class='gap-card {gap_class}'>"
        f"<div class='gap-numbers'>"
        f"<div class='gap-col'><div class='gap-label'>Listed</div><div class='gap-value'>{listing_price:.0f}L</div></div>"
        f"<div class='gap-arrow'>&rarr;</div>"
        f"<div class='gap-col'><div class='gap-label'>Estimated</div><div class='gap-value'>{pred_median:.0f}L</div></div>"
        f"<div class='gap-arrow'>=</div>"
        f"<div class='gap-col'><div class='gap-label'>Gap</div><div class='gap-value gap-diff'>{sign}{gap:.0f}L</div></div>"
        f"</div>"
        f"<div class='gap-msg'>{gap_icon} {gap_msg}</div>"
        f"</div>"
    )


def _build_sentence_explanation(contributions, pred_median):
    """C: Natural language explanation from top contributions."""
    positives = [(l, v) for l, v in contributions if v > 0]
    negatives = [(l, v) for l, v in contributions if v < 0]

    parts = []
    parts.append(f"This property is estimated at <strong>{pred_median:.0f}L</strong>.")

    if positives:
        top_pos = positives[:3]
        drivers = ', '.join(f"{l} (+{v:.0f}L)" for l, v in top_pos)
        parts.append(f"The main price drivers are <strong>{drivers}</strong>.")

    if negatives:
        top_neg = negatives[:2]
        reducers = ', '.join(f"{l} ({v:.0f}L)" for l, v in top_neg)
        parts.append(f"Factors pulling the price down: <strong>{reducers}</strong>.")

    return f"<div class='explain-sentence'>{' '.join(parts)}</div>"


def _build_contrib_table(contributions, pred_median):
    """B: Clean +/- table of all feature contributions."""
    rows = ""
    for label, lakhs in contributions:
        if lakhs >= 0:
            badge_class = 'contrib-pos'
            sign = '+'
        else:
            badge_class = 'contrib-neg'
            sign = ''
        rows += (
            f"<tr>"
            f"<td class='contrib-feature'>{label}</td>"
            f"<td class='contrib-impact'><span class='contrib-badge {badge_class}'>{sign}{lakhs:.1f}L</span></td>"
            f"</tr>"
        )

    return (
        f"<div class='contrib-section'>"
        f"<div class='contrib-title'>Feature Breakdown</div>"
        f"<table class='contrib-table'>"
        f"<tr class='contrib-header'><td>Feature</td><td>Impact</td></tr>"
        f"{rows}"
        f"<tr class='contrib-footer'>"
        f"<td class='contrib-feature'>Estimated Price</td>"
        f"<td class='contrib-impact'><span class='contrib-badge contrib-total'>{pred_median:.1f}L</span></td>"
        f"</tr>"
        f"</table>"
        f"</div>"
    )


def _build_stacked_waterfall(contributions, pred_median):
    """A: True stacking waterfall — each bar starts where the previous ended.

    Contributions now sum to pred_median (no separate baseline), so
    the waterfall builds from 0 to estimated price.
    """
    # Walk contributions, tracking running total from 0.
    running = 0.0
    steps = []
    for label, lakhs in contributions:
        start = running
        running += lakhs
        steps.append((label, start, running, lakhs))

    # Axis: 0 to a bit beyond the max position seen
    all_vals = [0.0] + [s[1] for s in steps] + [s[2] for s in steps]
    axis_min = min(all_vals)
    axis_max = max(all_vals) * 1.08
    axis_span = axis_max - axis_min
    if axis_span < 1:
        axis_span = 1

    def to_pct(val):
        return ((val - axis_min) / axis_span) * 100

    rows = ""
    for i, (label, start, end, lakhs_val) in enumerate(steps):
        left = to_pct(min(start, end))
        width = abs(to_pct(end) - to_pct(start))
        width = max(width, 0.5)

        sign = '+' if lakhs_val >= 0 else ''
        val_text = f"{sign}{lakhs_val:.0f}L"
        bar_class = 'sw-bar-pos' if lakhs_val >= 0 else 'sw-bar-neg'
        val_class = 'sw-val-pos' if lakhs_val >= 0 else 'sw-val-neg'

        connector = ""
        if i > 0:
            prev_end_pct = to_pct(steps[i - 1][2])
            connector = f"<div class='sw-connector' style='left:{prev_end_pct:.1f}%'></div>"

        rows += (
            f"<div class='sw-row'>"
            f"<div class='sw-label'>{label}</div>"
            f"<div class='sw-track'>"
            f"{connector}"
            f"<div class='sw-bar {bar_class}' style='left:{left:.1f}%;width:{width:.1f}%'></div>"
            f"</div>"
            f"<div class='sw-val {val_class}'>{val_text}</div>"
            f"</div>"
        )

    # Total row — full bar from 0 to estimated price
    rows += (
        f"<div class='sw-row sw-total-row'>"
        f"<div class='sw-label'>Estimated</div>"
        f"<div class='sw-track'>"
        f"<div class='sw-bar sw-bar-total' style='left:{to_pct(0):.1f}%;width:{to_pct(pred_median) - to_pct(0):.1f}%'></div>"
        f"</div>"
        f"<div class='sw-val sw-val-total'>{pred_median:.0f}L</div>"
        f"</div>"
    )

    return (
        f"<div class='sw-section'>"
        f"<div class='sw-title'>Price Build-Up</div>"
        f"<div class='sw-chart'>{rows}</div>"
        f"</div>"
    )


def _build_price_analysis(features_scaled, listing_price, pred_lower, pred_median, pred_upper):
    """Build all four explanation components for a property."""
    contributions = _get_contributions(features_scaled, pred_median)

    gap_card = _build_gap_card(listing_price, pred_median, pred_lower, pred_upper)
    sentence = _build_sentence_explanation(contributions, pred_median)
    waterfall = _build_stacked_waterfall(contributions, pred_median)

    return f"{gap_card}{sentence}{waterfall}"


def _build_verdict(listing_price, pred_lower, pred_median, pred_upper):
    """Return (css_class, short_label) for a listing price vs predicted range."""
    range_width = pred_upper - pred_lower
    if listing_price < pred_lower:
        return 'tag-good', 'Good Deal'
    elif listing_price > pred_upper:
        return 'tag-overpriced', 'Overpriced'
    elif listing_price > pred_upper - range_width * 0.25:
        return 'tag-caution', 'Near Upper End'
    else:
        return 'tag-fair', 'Fair Price'


def _build_range_bar(listing_price, pred_lower, pred_median, pred_upper):
    """Build range bar HTML snippet."""
    bar_min = pred_lower * 0.85
    bar_max = pred_upper * 1.15
    bar_span = bar_max - bar_min
    lower_pct = ((pred_lower - bar_min) / bar_span) * 100
    upper_pct = ((pred_upper - bar_min) / bar_span) * 100
    median_pct = ((pred_median - bar_min) / bar_span) * 100
    marker_pct = max(0, min(100, ((listing_price - bar_min) / bar_span) * 100))

    return (
        f"<div class='range-bar-container'><div class='range-bar-track'>"
        f"<div class='range-bar-fill' style='left:{lower_pct:.1f}%;width:{upper_pct - lower_pct:.1f}%;'></div>"
        f"<div class='range-bar-median' style='left:{median_pct:.1f}%;'></div>"
        f"<div class='range-bar-marker' style='left:{marker_pct:.1f}%;'></div>"
        f"</div><div class='range-bar-labels'>"
        f"<span>{pred_lower:.1f}L</span>"
        f"<span style='position:absolute;left:{median_pct:.1f}%;transform:translateX(-50%)'>{pred_median:.1f}L</span>"
        f"<span>{pred_upper:.1f}L</span>"
        f"</div></div>"
    )


# --- Pre-compute catalog data ---
def _load_catalog():
    """Load dataset, run predictions on every row, return list of dicts."""
    raw = pd.read_excel(
        os.path.join(os.path.dirname(__file__), 'Pune_Real_Estate_Data.xlsx')
    )
    catalog = []
    for _, row in raw.iterrows():
        # Parse price
        try:
            price = float(str(row['Price in lakhs']).strip())
        except (ValueError, TypeError):
            continue

        area_raw = str(row['Property Area in Sq. Ft.']).strip()
        # Take midpoint if range
        nums = _re.findall(r'[\d.]+', area_raw)
        if not nums:
            continue
        area_sqft = sum(float(n) for n in nums) / len(nums)

        beds = _extract_bedrooms(row['Propert Type'])
        sub = normalize_text_value(row['Sub-Area'])
        company = normalize_text_value(row['Company Name'])
        desc = normalize_text_value(row.get('Description', ''))
        township = str(row.get('TownShip Name/ Society Name', '') or '').strip()

        pool = 1 if _yn(row.get('Swimming Pool', 'no')) else 0
        club = 1 if _yn(row.get('ClubHouse', 'no')) else 0
        park = 1 if _yn(row.get('Park / Jogging track', 'no')) else 0
        gym = 1 if _yn(row.get('Gym', 'no')) else 0
        mall_v = 1 if _yn(row.get('Mall in TownShip', 'no')) else 0
        hospital_v = 1 if _yn(row.get('Hospital in TownShip', 'no')) else 0
        school_v = 1 if _yn(row.get('School / University in Township ', 'no')) else 0

        neighbourhood_amenities = mall_v + hospital_v + school_v
        property_amenities = club + park + gym
        desc_signal = compute_desc_price_signal(desc)
        area_log = np.log1p(area_sqft)

        company_encoded = te_company.transform(pd.DataFrame({'company_name': [company]}))[0][0]
        sub_area_encoded = te_sub_area.transform(pd.DataFrame({'sub_area': [sub]}))[0][0]

        features = pd.DataFrame([{
            'swimming_pool': pool,
            'no_of_bedrooms': beds,
            'neighbourhood_amenities': neighbourhood_amenities,
            'property_amenities': property_amenities,
            'desc_price_signal': desc_signal,
            'property_area_log': area_log,
            'te_company_name': company_encoded,
            'te_sub_area': sub_area_encoded,
        }]).reindex(columns=model_feature_columns, fill_value=0)

        features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

        pred_lower = float(np.expm1(qr_lower.predict(features_scaled)[0] - q_hat))
        pred_median = float(np.expm1(qr_median.predict(features_scaled)[0]))
        pred_upper = float(np.expm1(qr_upper.predict(features_scaled)[0] + q_hat))

        amenity_icons = []
        if pool:
            amenity_icons.append('Pool')
        if club:
            amenity_icons.append('Club')
        if park:
            amenity_icons.append('Park')
        if gym:
            amenity_icons.append('Gym')
        if mall_v:
            amenity_icons.append('Mall')
        if hospital_v:
            amenity_icons.append('Hospital')
        if school_v:
            amenity_icons.append('School')

        catalog.append({
            'price': price,
            'area': area_sqft,
            'beds': beds,
            'sub_area': sub,
            'company': company,
            'township': township,
            'desc': desc,
            'amenities': amenity_icons,
            'pred_lower': pred_lower,
            'pred_median': pred_median,
            'pred_upper': pred_upper,
            'features_scaled': features_scaled,
            'property_type': str(row['Propert Type']).strip(),
        })

    return catalog


CATALOG = _load_catalog()

# Collect unique sub-areas and BHK types for filters
CATALOG_SUB_AREAS = sorted(set(p['sub_area'] for p in CATALOG))
CATALOG_BHKS = sorted(set(p['property_type'].upper() for p in CATALOG))


def build_catalog_html(sub_area_filter, bhk_filter, sort_by):
    """Build the full catalog HTML from pre-computed data."""
    items = CATALOG

    # Filter
    if sub_area_filter and sub_area_filter != 'All':
        items = [p for p in items if p['sub_area'] == normalize_text_value(sub_area_filter)]
    if bhk_filter and bhk_filter != 'All':
        items = [p for p in items if p['property_type'].upper() == bhk_filter.upper()]

    # Sort
    if sort_by == 'Price: Low to High':
        items = sorted(items, key=lambda p: p['price'])
    elif sort_by == 'Price: High to Low':
        items = sorted(items, key=lambda p: p['price'], reverse=True)
    elif sort_by == 'Best Deals First':
        items = sorted(items, key=lambda p: p['price'] - p['pred_median'])
    else:
        items = sorted(items, key=lambda p: p['area'])

    if not items:
        return "<div class='empty-state'><div class='empty-state-text'>No properties match your filters.</div></div>"

    cards = []
    for i, p in enumerate(items):
        tag_class, tag_label = _build_verdict(p['price'], p['pred_lower'], p['pred_median'], p['pred_upper'])
        analysis = _build_price_analysis(p['features_scaled'], p['price'], p['pred_lower'], p['pred_median'], p['pred_upper'])

        amenity_tags = ''.join(f"<span class='amenity-tag'>{a}</span>" for a in p['amenities'])

        desc_short = _html.escape(p['desc'][:150] + ('...' if len(p['desc']) > 150 else '')) if p['desc'] else ''

        card = f"""<div class='catalog-card' onclick="this.classList.toggle('expanded')">
  <div class='card-header'>
    <div class='card-main'>
      <div class='card-title-row'>
        <span class='card-title'>{_html.escape(p['township'] or p['company'].title())}</span>
        <span class='card-tag {tag_class}'>{tag_label}</span>
      </div>
      <div class='card-subtitle'>{p['sub_area'].title()} &middot; {p['property_type'].upper()} &middot; {p['area']:.0f} sq ft</div>
      <div class='card-amenities'>{amenity_tags}</div>
    </div>
    <div class='card-price-col'>
      <div class='card-listed-price'>{p['price']:.0f}L</div>
      <div class='card-est-price'>Est. {p['pred_median']:.0f}L</div>
    </div>
  </div>
  <div class='card-expand-hint'>Click to see price breakdown</div>
  <div class='card-detail'>
    <div class='card-desc'>{desc_short}</div>
    {analysis}
  </div>
</div>"""
        cards.append(card)

    count_html = f"<div class='catalog-count'>{len(items)} properties</div>"
    return count_html + "\n".join(cards)


# --- Load CSS from file ---
with open(os.path.join(os.path.dirname(__file__), 'style.css')) as f:
    CUSTOM_CSS = f.read()

# --- Gradio UI ---
EMPTY_STATE = (
    "<div class='empty-state'>"
    "<svg width='48' height='48' viewBox='0 0 24 24' fill='none' stroke='#b0c4de' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round' style='margin-bottom: 12px;'>"
    "<path d='M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/>"
    "<polyline points='9 22 9 12 15 12 15 22'/>"
    "</svg>"
    "<div class='empty-state-text'>Fill in property details and choose an action</div>"
    "</div>"
)

with gr.Blocks(title="Pillow: Price Discovery for Pune Real Estate") as app:
    gr.HTML("<link rel='preconnect' href='https://fonts.googleapis.com'><link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>")

    gr.HTML("<h1 style='font-family: Inter, sans-serif; font-size: 2.2rem; font-weight: 700; text-align: center; color: #1d4ed8; margin: 0 0 4px;'>Pillow</h1>")
    gr.HTML("<p style='font-family: Inter, sans-serif; font-size: 1rem; text-align: center; color: #666; margin: 0 0 24px;'>Estimate fair value for Pune real estate</p>")

    with gr.Tabs(elem_classes=["pillow-tabs"]):

        # ==================== SELLER TAB ====================
        with gr.Tab("Seller", elem_id="seller-tab"):
            with gr.Row(elem_classes=["main-layout"]):
                with gr.Column(scale=3, elem_classes=["input-column"]):
                    with gr.Column(elem_classes=["card-panel"]):
                        gr.HTML("<div class='step-header'><span class='step-badge'>1</span><div class='section-title'><svg viewBox='0 0 24 24'><path d='M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/><polyline points='9 22 9 12 15 12 15 22'/></svg>Property Details</div></div>")
                        area_sqft = gr.Slider(200, 3000, value=900, step=1, label="Property Area (sq ft)")
                        bedrooms = gr.Slider(1, 6, value=2, step=0.5, label="Number of Bedrooms")
                        with gr.Row():
                            company_name = gr.Dropdown(COMPANIES, label="Builder / Company", value='mantra properties', elem_classes=["styled-dropdown"])
                            sub_area = gr.Dropdown(SUB_AREAS, label="Sub Area", value='baner', elem_classes=["styled-dropdown"])

                    with gr.Column(elem_classes=["card-panel"]):
                        gr.HTML("<div class='step-header'><span class='step-badge'>2</span><div class='section-title'><svg viewBox='0 0 24 24'><rect x='3' y='3' width='7' height='7'/><rect x='14' y='3' width='7' height='7'/><rect x='14' y='14' width='7' height='7'/><rect x='3' y='14' width='7' height='7'/></svg>Amenities</div></div>")
                        with gr.Row(equal_height=True, elem_classes=["amenity-row"]):
                            swimming_pool = gr.Checkbox(label="Swimming Pool")
                            clubhouse = gr.Checkbox(label="Clubhouse")
                            park_jogging_track = gr.Checkbox(label="Park / Jogging Track")
                            gym_check = gr.Checkbox(label="Gym")
                        with gr.Row(equal_height=True, elem_classes=["amenity-row"]):
                            school = gr.Checkbox(label="School / University")
                            mall = gr.Checkbox(label="Mall in Township")
                            hospital = gr.Checkbox(label="Hospital in Township")
                            select_all = gr.Checkbox(label="Select All")

                    with gr.Column(elem_classes=["card-panel"]):
                        gr.HTML("<div class='step-header'><span class='step-badge'>3</span><div class='section-title'><svg viewBox='0 0 24 24'><path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/><polyline points='14 2 14 8 20 8'/><line x1='16' y1='13' x2='8' y2='13'/><line x1='16' y1='17' x2='8' y2='17'/><polyline points='10 9 9 9 8 9'/></svg>Description</div></div>")
                        description = gr.Textbox(
                            label="Property Description",
                            placeholder="This is a luxurious property boasting 4.5 bedrooms in a tall building with a river front view.",
                            lines=3,
                        )

                    with gr.Column(elem_classes=["card-panel"]):
                        gr.HTML("<div class='step-header'><span class='step-badge'>4</span><div class='section-title'><svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10'/><polyline points='12 6 12 12 16 14'/></svg>Choose an Action</div></div>")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, elem_classes=["action-card"]):
                                gr.HTML("<div class='action-label'>Get the model's estimated price range</div>")
                                gr.HTML("<div class='action-spacer'></div>")
                                estimate_btn = gr.Button("Estimate Range", variant="primary", elem_classes=["gen-btn"])

                            with gr.Column(scale=1, elem_classes=["action-card"]):
                                gr.HTML("<div class='action-label'>Enter your price and see if it's fair</div>")
                                listing_price = gr.Number(
                                    label="Your Price (in lakhs)",
                                    value=None,
                                    minimum=0,
                                )
                                check_btn = gr.Button("Check My Price", variant="secondary", elem_classes=["gen-btn", "check-btn"])

                with gr.Column(scale=2, elem_classes=["result-column"]):
                    with gr.Column(elem_classes=["result-panel"]):
                        gr.HTML("<div class='section-title' style='justify-content: center;'><svg viewBox='0 0 24 24'><line x1='12' y1='1' x2='12' y2='23'/><path d='M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6'/></svg>Estimated Price Range</div>")
                        output = gr.HTML(value=EMPTY_STATE)

        # ==================== BUYER TAB ====================
        with gr.Tab("Buyer", elem_id="buyer-tab"):
            with gr.Row(elem_classes=["filter-bar"]):
                b_sub_filter = gr.Dropdown(
                    ['All'] + CATALOG_SUB_AREAS,
                    label="Location", value='All',
                    elem_classes=["styled-dropdown", "filter-dropdown"],
                )
                b_bhk_filter = gr.Dropdown(
                    ['All'] + CATALOG_BHKS,
                    label="Property Type", value='All',
                    elem_classes=["styled-dropdown", "filter-dropdown"],
                )
                b_sort = gr.Dropdown(
                    ['Best Deals First', 'Price: Low to High', 'Price: High to Low', 'Area'],
                    label="Sort By", value='Best Deals First',
                    elem_classes=["styled-dropdown", "filter-dropdown"],
                )

            catalog_output = gr.HTML(
                value=build_catalog_html('All', 'All', 'Best Deals First'),
                elem_classes=["catalog-container"],
            )

    # --- Seller tab wiring ---
    amenity_checkboxes = [swimming_pool, clubhouse, park_jogging_track, gym_check, school, mall, hospital]

    def toggle_select_all(checked):
        if checked:
            return [gr.update(value=True, interactive=False) for _ in amenity_checkboxes]
        else:
            return [gr.update(value=False, interactive=True) for _ in amenity_checkboxes]

    select_all.change(
        fn=toggle_select_all,
        inputs=[select_all],
        outputs=amenity_checkboxes,
    )

    estimate_btn.click(
        fn=lambda *args: predict(*args, listing_price=None),
        inputs=[area_sqft, bedrooms, swimming_pool, mall, hospital, school,
                clubhouse, park_jogging_track, gym_check, company_name, sub_area,
                description],
        outputs=output,
    )

    check_btn.click(
        fn=predict,
        inputs=[area_sqft, bedrooms, swimming_pool, mall, hospital, school,
                clubhouse, park_jogging_track, gym_check, company_name, sub_area,
                description, listing_price],
        outputs=output,
    )

    # --- Buyer tab wiring ---
    buyer_filter_inputs = [b_sub_filter, b_bhk_filter, b_sort]
    for filt in buyer_filter_inputs:
        filt.change(
            fn=build_catalog_html,
            inputs=buyer_filter_inputs,
            outputs=catalog_output,
        )

    gr.HTML(
        "<div class='footer'>"
        "<a href='https://linkedin.com/in/aryandeore' target='_blank'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'/></svg>"
        "aryandeore"
        "</a>"
        "<a href='https://www.aryandeore.ai' target='_blank'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm-1 4.062V8H7.062A8.006 8.006 0 0 1 11 4.062zM4.062 13H8v3H5.674A7.953 7.953 0 0 1 4.062 13zm1.612 5H8v2.938A8.006 8.006 0 0 1 5.674 18zM11 19.938V16h3.326A8.006 8.006 0 0 1 11 19.938zM11 14v-3h2v3h-2zm0-5V4.062A8.006 8.006 0 0 1 14.938 8H11zm5 9v-3h2.326a7.953 7.953 0 0 1-2.326 3zm2.326-5H16v-3h3.938a7.953 7.953 0 0 1-1.612 3zM16 8V5.062A8.006 8.006 0 0 1 19.938 9H16zm-3-3.938V8h-2V4.062A8.006 8.006 0 0 1 13 4.062z'/></svg>"
        "aryandeore.ai"
        "</a>"
        "<a href='mailto:aryandeore.work@gmail.com'>"
        "<svg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path d='M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-.9.732-1.636 1.636-1.636h.749L12 10.638 21.615 3.82h.749c.904 0 1.636.737 1.636 1.636z'/></svg>"
        "Get in touch"
        "</a>"
        "</div>"
    )

if __name__ == '__main__':
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CUSTOM_CSS,
    )
