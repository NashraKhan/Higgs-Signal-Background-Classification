# Grok UI connection patch

The Grok frontend remains in the separate Higgs-Grok-Dashboard-Review folder.
This branch contains a replacement for its Chamber shell at:

    grok-dashboard-patch/src/components/chamber/shell.tsx

Copy that file into the matching path in the Grok frontend folder. It replaces the
random "Release a bunch" outcome with a real call to the local Python API.

The frontend defaults to http://127.0.0.1:8000. See higgs_inference/README.md for
starting the API and providing the held-out test data.
