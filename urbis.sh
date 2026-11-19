#!/bin/sh
# Start the URBIS web app on this computer and open it: http://localhost:8770
cd "$(dirname "$0")"
(sleep 2; open http://localhost:8770 2>/dev/null || xdg-open http://localhost:8770 2>/dev/null) &
exec .venv/bin/python scripts/urbis_app.py 8770
