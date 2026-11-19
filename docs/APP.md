# URBIS web app

Map the slums and informal settlements of any city, then count and number every house. Runs on this computer.

## Start

```bash
./urbis.sh
```

This opens http://localhost:8770. To stop it, press Ctrl-C in the terminal. Equivalent: `.venv/bin/python scripts/urbis_app.py 8770`.

## Use

1. **Landing page → Create account.** Accounts are local to this computer, stored in `runs/app/users.json` with scrypt-hashed passwords. Each user sees only their own cities.
2. **Your cities → New city.**
   1. Search a place (OpenStreetMap Nominatim). The box defaults to the place's area, at most 14 × 14 km.
   2. Drag the white corner handles, or click "draw a new box" and drag on the map. Limits: 1 to 600 km².
   3. Name the city and press Create city.
3. **Set-up runs in the background,** one city at a time; the page shows progress. Time is about 6 min plus 0.3 min per km² on this Mac (the Kisumu test, 20 km², took 12 min).
   1. 100 m grid (`data/<id>/labels.tif`).
   2. Overture building footprints for the box (`runs/pipeline/<id>/buildings.parquet`).
   3. The round-3 slum model scores every 100 m box that has a building (`runs/app_cities/<id>.npz`).
4. **Layers → Model's first guess** shows the model's top 5 % boxes (pseudo-labels) before anyone reviews.
5. **Review.** Boxes come most-likely first; answer Slum (Y), Not slum (N) or Skip (S); Z zooms to the box. The order adapts to the answers.
   - Each box shows its model rank.
   - Review status says **Review complete** when URBIS's internal estimate of slum still unfound drops under 3 %. On 23 test cities the stop rule fired after about 23 % of built-up boxes (22–33 %) and every city reached its target. Most slum is found much earlier: after 10 % of built-up boxes, 99.5 % of the slum area in settlements of 3 ha or more was found (20 of 23 cities at 99 % or more).
6. **Rebuild outlines** (Review tab, about 1–3 min). The AI draws settlement outlines from the confirmed boxes, then lists every house inside them, numbered within its settlement.
   - Overview shows settlements, area, houses and density.
   - On the map, houses appear from zoom 16 and their numbers from zoom 19.
7. **Export:** GeoJSON, KML (Google Earth), GeoPackage (2 layers), Shapefile zip, FlatGeobuf zip, settlements CSV, houses CSV (WKT), 10 m GeoTIFF, HTML report.

## Files

| What | Where |
|---|---|
| Server (stdlib HTTP, auth, jobs, exports) | `scripts/urbis_app.py` |
| New-city set-up (grid, buildings, scoring) | `scripts/new_city.py` |
| Review order, stop rule | `scripts/city_pipeline.py` |
| Outlines and houses | `scripts/city_outlines.py` (model `runs/kaggle_outline/out1/mvit.pt`) |
| Slum model | `runs/kaggle_retrain3/data/visible_verifier_sat_ibge.pt` + `runs/kaggle_retrain3/out/twoview_e2.pt`; DINOv3 ViT-L sat in the Hugging Face cache |
| Pages | `app/` (`index.html` landing, `login.html`, `cities.html`, `new.html`, `city.html` + `app.js`, `common.js`, `app.css`) |
| App data | `runs/app/` (`users.json`, `sessions.json`, `cities.json`), answers in `runs/pipeline/<id>/answers.csv` |

## Limits (plain)

- **Houses** come from footprint maps (Google, Microsoft, OSM). Tightly packed roofs can merge into one footprint, so counts are a minimum. The URBIS roof model (`scripts/house_train.py`) is not good enough yet: house F1 0.09 in Accra and 0.21 in Kampala against a 0.6 target. The app uses it automatically for a city once `runs/pipeline/<id>/houses_ai.geojson` exists (`scripts/city_houses.py city <id> <model>`).
- **The model's ranking is a starting point, not a map.** On cities it never saw, about 38 % of its top marks are real slums when it finds 80 % of them. The human review is what makes the map reliable.
- **Local-only server** (binds 127.0.0.1). Before putting it on a network, add HTTPS and a real database, and review the sign-up policy.
- **One set-up runs at a time;** others wait in a queue that survives a restart.
- The scoring step needs a GPU or Apple MPS to be fast. On CPU, expect several times longer.
