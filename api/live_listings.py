"""Live listings service — parse today's HTML files via `src/parser.py`, batch-score via Phase-7 bundle, cache to CSV."""
from __future__ import annotations

import glob
import math
import os
from datetime import date, timedelta

import pandas as pd

from api.config import PROJECT_DIR

CACHE_CSV = os.path.join(PROJECT_DIR, "outputs", "today_predictions.csv")
DATA_DIR = os.path.join(PROJECT_DIR, "data")


class LiveListingsService:
    def __init__(self, prediction_service, data_loader):
        self.prediction_service = prediction_service
        self.data_loader = data_loader
        self.listings_df = None
        self.date_used = None
        self.load_or_generate()

    def load_or_generate(self):
        """Load from CSV cache if today's data exists, otherwise parse + predict."""
        if os.path.exists(CACHE_CSV):
            df = pd.read_csv(CACHE_CSV)
            if "date_obtained" in df.columns and len(df) > 0:
                cached_date = str(df["date_obtained"].iloc[0])
                if cached_date == str(date.today()):
                    self.listings_df = df
                    self.date_used = cached_date
                    print(f"Live listings: loaded {len(df)} from cache ({cached_date})")
                    return

        self._generate()

    def refresh(self):
        """Force re-parse + re-predict, ignoring cache."""
        self._generate()
        return {"count": len(self.listings_df), "date_used": self.date_used}

    def _generate(self):
        """Scan HTML files, parse with canonical `src/parser.py`, batch-score with Phase-7 bundle, cache CSV."""
        html_files, date_tag = self._find_html_files()
        if not html_files:
            print("Live listings: no HTML files found in last 7 days")
            self.listings_df = pd.DataFrame()
            self.date_used = None
            return

        if self.prediction_service is None or getattr(self.prediction_service, "bundle", None) is None:
            print("Live listings: production bundle not loaded; skipping scoring")
            self.listings_df = pd.DataFrame()
            self.date_used = None
            return

        iso_date = f"{date_tag[4:8]}-{date_tag[2:4]}-{date_tag[0:2]}"
        self.date_used = iso_date
        total = len(html_files)
        print(f"Live listings: processing {total} files from {iso_date}...")

        parsed_rows = []
        meta_rows = []
        parse_errors = 0

        for i, (html_path, property_id) in enumerate(html_files):
            if (i + 1) % 100 == 0:
                print(f"  Parsing {i + 1}/{total}...")
            try:
                with open(html_path, "r", encoding="utf-8", errors="replace") as f:
                    html = f.read()
                fake_url = f"https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/9000/{property_id}"
                parsed = self.prediction_service.parse_immoweb_html(html, fake_url)
                asking = parsed.get("price")
                area = parsed.get("livable_area")
                if not asking or not area or float(asking) <= 0 or float(area) <= 0:
                    continue
                parsed_rows.append(parsed)

                street = str(parsed.get("street") or "").strip()
                house_num = str(parsed.get("house_number") or "").strip()
                postal = str(parsed.get("postal_code") or "").strip()
                city = str(parsed.get("city") or "").strip()
                addr_left = f"{street} {house_num}".strip()
                addr_right = f"{postal} {city}".strip()
                address = ", ".join(p for p in (addr_left, addr_right) if p)

                meta_rows.append({
                    "property_id": str(property_id),
                    "address": address,
                    "url": fake_url,
                    "asking_price_eur": float(asking),
                    "livable_area": float(area),
                    "bedrooms": parsed.get("number_of_bedrooms"),
                    "condition": parsed.get("building_condition"),
                    "energy_label": parsed.get("energy_label"),
                    "property_type": parsed.get("property_type"),
                    "latitude": parsed.get("latitude"),
                    "longitude": parsed.get("longitude"),
                })
            except Exception:
                parse_errors += 1

        if not parsed_rows:
            print(f"Live listings: no valid properties parsed ({parse_errors} errors)")
            self.listings_df = pd.DataFrame()
            return

        print(f"  Parsed {len(parsed_rows)} properties ({parse_errors} errors). Batch scoring...")
        pred_out = self.prediction_service.predict_batch_parsed(parsed_rows)
        feat_df = pred_out["feat_df"]

        output_rows = []
        for i, meta in enumerate(meta_rows):
            predicted_eur = float(pred_out["predicted_eur"][i])
            low_eur = float(pred_out["low_eur"][i])
            high_eur = float(pred_out["high_eur"][i])
            hw_pct = float(pred_out["interval_halfwidth_pct"][i])
            asking_eur = meta["asking_price_eur"]
            delta_eur = predicted_eur - asking_eur
            delta_pct = (delta_eur / asking_eur * 100.0) if asking_eur else 0.0
            if asking_eur < low_eur:
                band_position = "below_band"
            elif asking_eur > high_eur:
                band_position = "above_band"
            else:
                band_position = "inside_band"

            nbh = None
            if "neighbourhood" in feat_df.columns and pd.notna(feat_df["neighbourhood"].iloc[i]):
                cand = str(feat_df["neighbourhood"].iloc[i])
                if cand and cand != "Unknown":
                    nbh = cand
            if not nbh and meta["latitude"] and meta["longitude"] and self.data_loader:
                nbh = self.data_loader.get_neighbourhood_spatial(float(meta["latitude"]), float(meta["longitude"]))

            output_rows.append({
                "property_id": meta["property_id"],
                "address": meta["address"],
                "url": meta["url"],
                "asking_price": round(asking_eur),
                "predicted_price": round(predicted_eur),
                "low_eur": round(low_eur),
                "high_eur": round(high_eur),
                "interval_halfwidth_pct": round(hw_pct, 1),
                "band_position": band_position,
                "delta_eur": round(delta_eur),
                "delta_pct": round(float(delta_pct), 1),
                "livable_area": meta["livable_area"],
                "bedrooms": meta["bedrooms"],
                "condition": meta["condition"],
                "energy_label": meta["energy_label"],
                "property_type": meta["property_type"],
                "neighbourhood": nbh,
                "date_obtained": iso_date,
                "source": "live",
            })

        self.listings_df = pd.DataFrame(output_rows)
        os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
        self.listings_df.to_csv(CACHE_CSV, index=False)
        print(f"Live listings: scored & cached {len(self.listings_df)} listings ({iso_date})")

    def _find_html_files(self):
        """Find HTML files from today or the most recent date within the last 7 days."""
        today = date.today()
        for days_ago in range(7):
            d = today - timedelta(days=days_ago)
            date_tag = d.strftime("%d%m%Y")
            pattern = os.path.join(DATA_DIR, "*", "*", f"*_{date_tag}.html")
            matches = glob.glob(pattern)
            if matches:
                seen = {}
                for path in matches:
                    basename = os.path.basename(path)
                    pid = basename.rsplit("_", 1)[0]
                    seen[pid] = path
                return [(path, pid) for pid, path in seen.items()], date_tag
        return [], None

    def get_listings(self, neighbourhood=None):
        """Return list of live listing dicts, optionally filtered by neighbourhood."""
        if self.listings_df is None or len(self.listings_df) == 0:
            return []

        df = self.listings_df
        if neighbourhood:
            def _normalize(s):
                return str(s).lower().replace("_", " ").replace("-", " ").strip()
            target = _normalize(neighbourhood)
            mask = df["neighbourhood"].dropna().apply(lambda x: _normalize(x) == target)
            df = df.loc[mask[mask].index]

        df = df.sort_values("delta_pct", ascending=False)
        records = df.to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
        return records

    def get_deals(
        self,
        *,
        neighbourhood: str | None = None,
        property_type: str | None = None,
        max_price: float | None = None,
        max_predicted: float | None = None,
        min_epc: str | None = None,
        min_condition: str | None = None,
        min_area: float | None = None,
        min_bedrooms: int | None = None,
        regime: str | None = None,
        has_garden: bool = False,
        has_terrace: bool = False,
        price_cut_only: bool = False,
        min_jepa_dz: float | None = None,
        search: str | None = None,
        include_archive: bool = False,
        min_delta_pct: float | None = None,
        below_band_only: bool = False,
        available_only: bool = False,
        limit: int = 200,
    ) -> dict:
        """City-wide deal feed: uses live scraped listings when fresh, otherwise falls back
        to local Phase-7 OOF scored corpus listings (no Immoweb requests needed).
        """
        EPC_RANK = {"F": 0, "E": 1, "D": 2, "C": 3, "B": 4, "A": 5, "A+": 6, "A++": 7}
        COND_RANK = {
            "TO_RESTORE": 0,
            "TO_RENOVATE": 1,
            "TO_REFURBISH": 1,
            "TO_BE_DONE_UP": 2,
            "GOOD": 3,
            "FIRST_TIME_USED": 4,
            "JUST_RENOVATED": 4,
            "AS_NEW": 5,
        }

        if not include_archive and self.listings_df is not None and len(self.listings_df) > 0:
            df = self.listings_df.copy()
            source = "live_scrape"
            date_used = self.date_used
            if "band_position" not in df.columns and {"asking_price", "low_eur", "high_eur"}.issubset(df.columns):
                df["band_position"] = df.apply(
                    lambda r: (
                        "below_band" if r["asking_price"] < r["low_eur"]
                        else ("above_band" if r["asking_price"] > r["high_eur"] else "inside_band")
                    ),
                    axis=1,
                )
            if "availability_status" not in df.columns:
                df["availability_status"] = "active_recent"
                df["last_seen"] = self.date_used
                df["days_since_last_seen"] = 0
                df["latest_scrape_date"] = self.date_used
        else:
            source = "corpus_full" if include_archive else "corpus_recent"
            date_used = None
            adf = getattr(self.data_loader, "analytics_df", None) if self.data_loader is not None else None
            if not isinstance(adf, pd.DataFrame) or adf.empty or "pred_mm_eur" not in adf.columns:
                return {"deals": [], "source": "empty", "date_used": None, "total": 0}

            sub = adf.copy()
            if "date_obtained" in sub.columns:
                sub["_dt"] = pd.to_datetime(sub["date_obtained"], errors="coerce")
                # Prefer last_seen_str if present so listings first seen earlier but STILL active
                # in the latest scrape stay included in the recent deal pool!
                if "last_seen_str" in sub.columns:
                    sub["_last_dt"] = pd.to_datetime(sub["last_seen_str"], errors="coerce").fillna(sub["_dt"])
                else:
                    sub["_last_dt"] = sub["_dt"]
                max_dt = sub["_last_dt"].max()
                if pd.notna(max_dt):
                    date_used = max_dt.strftime("%Y-%m-%d")
                    if not include_archive:
                        cutoff = max_dt - pd.Timedelta(days=90)
                        recent = sub[sub["_last_dt"] >= cutoff]
                        if len(recent) >= 100:
                            sub = recent
            if "normal_mask" in sub.columns:
                sub = sub[sub["normal_mask"].fillna(True).astype(bool)]

            # Deduplicate identical physical listings (same neighbourhood + asking + area + beds + EPC)
            dedup_cols = [c for c in ("neighbourhood", "price_eur", "livable_area", "number_of_bedrooms", "energy_label") if c in sub.columns]
            if len(dedup_cols) >= 3:
                sort_col = "_last_dt" if "_last_dt" in sub.columns else ("_dt" if "_dt" in sub.columns else "property_id")
                sub = sub.sort_values(sort_col).drop_duplicates(subset=dedup_cols, keep="last")

            asking = pd.to_numeric(sub["price_eur"] if "price_eur" in sub.columns else sub["price"] * 1000.0, errors="coerce")
            pred = pd.to_numeric(sub["pred_mm_eur"], errors="coerce")
            valid = asking.notna() & pred.notna() & (asking > 0) & (pred > 0)
            sub = sub.loc[valid].copy()
            asking = asking.loc[valid]
            pred = pred.loc[valid]

            # Calibrated ~14.5% median 80% conformal half-width on multimodal OOF
            hw_frac = 0.145
            low_eur = pred / (1.0 + hw_frac)
            high_eur = pred * (1.0 + hw_frac)
            delta_eur = pred - asking
            delta_pct = delta_eur / asking * 100.0
            band_pos = pd.Series("inside_band", index=sub.index)
            band_pos = band_pos.mask(asking < low_eur, "below_band")
            band_pos = band_pos.mask(asking > high_eur, "above_band")

            nbh_col = "neighbourhood_display" if "neighbourhood_display" in sub.columns else "neighbourhood"
            pids = sub["property_id"].astype(str)
            ptype = (
                sub["property_type_label"]
                if "property_type_label" in sub.columns
                else sub["property_type_HOUSE"].map(lambda x: "HOUSE" if int(x or 0) == 1 else "APARTMENT")
            )

            # Build human-readable street address when available
            def _make_addr(r):
                st = str(r.get("street") or "").strip()
                hn = str(r.get("house_number") or "").strip()
                pc = str(r.get("postal_code") or "9000").strip()
                nb = str(r.get(nbh_col) or "").strip()
                if st and st.lower() not in ("nan", "none"):
                    left = f"{st} {hn}".strip() if hn and hn.lower() not in ("nan", "none") else st
                    return f"{left}, {pc} Gent"
                return f"{nb} · {pc} Gent"

            # Availability status derived from latest scrape snapshot
            days_ago = pd.to_numeric(sub["days_since_last_seen"], errors="coerce").fillna(999).astype(int) if "days_since_last_seen" in sub.columns else pd.Series(999, index=sub.index)
            is_opt = sub["is_under_option_latest"].fillna(False).astype(bool) if "is_under_option_latest" in sub.columns else (sub["is_under_option"].fillna(False).astype(bool) if "is_under_option" in sub.columns else pd.Series(False, index=sub.index))
            is_sold = sub["is_sold_or_rented"].fillna(False).astype(bool) if "is_sold_or_rented" in sub.columns else pd.Series(False, index=sub.index)
            avail_status = pd.Series("likely_gone", index=sub.index)
            avail_status = avail_status.mask(days_ago <= 14, "seen_recently")
            avail_status = avail_status.mask(days_ago <= 3, "active_recent")
            avail_status = avail_status.mask(is_opt, "under_option")
            avail_status = avail_status.mask(is_sold, "sold_or_rented")

            def _col(name, default=None):
                return sub[name] if name in sub.columns else default

            def _bool_col(name):
                if name not in sub.columns:
                    return False
                return sub[name].fillna(False).astype(bool)

            def _num_col(name, ndigits=None):
                if name not in sub.columns:
                    return None
                s = pd.to_numeric(sub[name], errors="coerce")
                return s.round(ndigits) if ndigits is not None else s

            df = pd.DataFrame({
                "property_id": pids,
                "address": sub.apply(_make_addr, axis=1),
                "url": "https://www.immoweb.be/nl/zoekertje/property/te-koop/gent/9000/" + pids,
                "asking_price": asking.round().astype(int),
                "predicted_price": pred.round().astype(int),
                "low_eur": low_eur.round().astype(int),
                "high_eur": high_eur.round().astype(int),
                "interval_halfwidth_pct": 14.5,
                "band_position": band_pos,
                "delta_eur": delta_eur.round().astype(int),
                "delta_pct": delta_pct.round(1),
                "livable_area": _num_col("livable_area"),
                "property_size": _num_col("property_size"),
                "bedrooms": _num_col("number_of_bedrooms"),
                "bathrooms": _num_col("number_of_bathrooms"),
                "year_built": _num_col("year_built"),
                "condition": sub["building_condition"],
                "energy_label": sub["energy_label"],
                "primary_energy_consumption": _num_col("primary_energy_consumption"),
                "property_type": ptype,
                "property_subtype": _col("property_subtype"),
                "neighbourhood": sub[nbh_col],
                "neighbourhood_key": sub["neighbourhood"],
                "postal_code": _col("postal_code"),
                "dist_from_center": _num_col("dist_from_center", 2),
                "has_garden": _bool_col("has_garden"),
                "garden_surface": _num_col("garden_surface"),
                "has_terrace": _bool_col("has_terrace"),
                "terrace_surface": _num_col("terrace_surface"),
                "has_lift": _bool_col("has_lift"),
                "has_photovoltaic_panels": _bool_col("has_photovoltaic_panels"),
                "has_heat_pump": _bool_col("has_heat_pump"),
                "cadastral_income": _num_col("cadastral_income"),
                "date_obtained": sub["_dt"].dt.strftime("%Y-%m-%d") if "_dt" in sub.columns else None,
                "first_seen": _col("first_seen_str"),
                "last_seen": _col("last_seen_str"),
                "days_since_last_seen": days_ago,
                "latest_scrape_date": _col("latest_scrape_date", date_used),
                "availability_status": avail_status,
                "listing_duration_days": _num_col("listing_duration_days"),
                "n_snapshots": _num_col("n_snapshots"),
                "is_under_option": is_opt,
                "had_price_cut": _bool_col("had_price_cut"),
                "price_cut_eur": _num_col("price_cut_eur", 0),
                "price_cut_pct": _num_col("price_cut_pct", 1),
                "agency": _col("agency_clean"),
                "title": _col("title"),
                "description_snippet": _col("description_snippet"),
                "vision_delta_eur": _num_col("vision_delta_eur", 0),
                "vision_delta_pct": _num_col("vision_delta_pct", 1),
                "property_regime": sub["property_regime"].fillna("residential") if "property_regime" in sub.columns else "residential",
                "jepa_dz_quality": _num_col("jepa_dz_quality", 2),
                "jepa_energy_sym": _num_col("jepa_energy_sym", 2),
                "source": "corpus_recent",
            })

        # Apply user filters
        def _norm(s):
            return str(s).lower().replace("_", " ").replace("-", " ").replace("–", " ").strip()

        if neighbourhood and neighbourhood.lower() not in ("all", "any"):
            target = _norm(neighbourhood)
            m_nbh = df["neighbourhood"].fillna("").apply(lambda x: target in _norm(x))
            if "neighbourhood_key" in df.columns:
                m_nbh |= df["neighbourhood_key"].fillna("").apply(lambda x: target == _norm(x))
            df = df[m_nbh]

        if property_type and property_type.upper() in ("HOUSE", "APARTMENT"):
            df = df[df["property_type"].astype(str).str.upper() == property_type.upper()]

        if max_price is not None and max_price > 0:
            df = df[pd.to_numeric(df["asking_price"], errors="coerce") <= max_price]

        if max_predicted is not None and max_predicted > 0:
            df = df[pd.to_numeric(df["predicted_price"], errors="coerce") <= max_predicted]

        if min_epc and min_epc.upper().rstrip("+") in EPC_RANK:
            min_rank = EPC_RANK[min_epc.upper().rstrip("+")]
            df = df[df["energy_label"].astype(str).str.upper().map(EPC_RANK).fillna(-1) >= min_rank]

        if min_condition and min_condition.upper().rstrip("+") in COND_RANK:
            min_cr = COND_RANK[min_condition.upper().rstrip("+")]
            df = df[df["condition"].astype(str).str.upper().map(COND_RANK).fillna(-1) >= min_cr]

        if min_area is not None and min_area > 0:
            df = df[pd.to_numeric(df["livable_area"], errors="coerce") >= min_area]

        if min_bedrooms is not None and min_bedrooms > 0:
            df = df[pd.to_numeric(df["bedrooms"], errors="coerce") >= min_bedrooms]

        if regime and regime.lower() not in ("all", "any"):
            if "property_regime" in df.columns:
                if regime.lower() == "non_standard":
                    df = df[df["property_regime"].fillna("residential") != "residential"]
                else:
                    df = df[df["property_regime"].fillna("residential") == regime.lower()]

        if has_garden and "has_garden" in df.columns:
            df = df[df["has_garden"].fillna(False).astype(bool)]

        if has_terrace and "has_terrace" in df.columns:
            df = df[df["has_terrace"].fillna(False).astype(bool)]

        if price_cut_only and "had_price_cut" in df.columns:
            df = df[df["had_price_cut"].fillna(False).astype(bool)]

        if min_jepa_dz is not None and "jepa_dz_quality" in df.columns:
            df = df[pd.to_numeric(df["jepa_dz_quality"], errors="coerce") >= min_jepa_dz]

        if search and search.strip():
            q = search.strip().lower()
            cols_to_search = [c for c in ("address", "neighbourhood", "title", "description_snippet", "agency", "property_id") if c in df.columns]
            if cols_to_search:
                m_q = pd.Series(False, index=df.index)
                for col in cols_to_search:
                    m_q |= df[col].fillna("").astype(str).str.lower().str.contains(q, regex=False)
                df = df[m_q]

        if min_delta_pct is not None:
            df = df[pd.to_numeric(df["delta_pct"], errors="coerce") >= min_delta_pct]

        if below_band_only and "band_position" in df.columns:
            df = df[df["band_position"] == "below_band"]

        if available_only and "availability_status" in df.columns:
            df = df[df["availability_status"].isin(["active_recent", "seen_recently"])]

        df = df.sort_values("delta_pct", ascending=False)
        total_matched = len(df)
        records = df.head(max(1, min(int(limit), 500))).to_dict(orient="records")
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    rec[k] = None
        return {
            "deals": records,
            "total": total_matched,
            "source": source,
            "date_used": date_used,
        }

    def get_count(self):
        count = len(self.listings_df) if self.listings_df is not None else 0
        return {"count": count, "date_used": self.date_used}
