# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-048`
- winner mode: `learned_route`
- trace hash: `sha256-9c32a87b231e4d5848a772d9d1cb8d355e8b17c5c883fc0f1ca8776ef042ba2c`
- fixture hash: `sha256-66d4441e9cd89d5df06e129fcf70accf27e8123573950bf81a6f813e2979adc4`
- score hash: `sha256-3f26725744a709098662af4ce0e253a25e3d6f6be53eeef96e6928ccfe371b47`
- bundle hash: `sha256-9387f03245a6e9e20349b018f6340c4ef6d166ce8345f5dc8a801053d6fb004e`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | learned_route | 100 |
| 2 | vector_only | 100 |
| 3 | graph_prior_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/4
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 1 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2b9d9843197c6ea9cf1bbaf94c65647f4ecfa1e2224f8678711a552cc896cd7e |
| vector_only | 1 | 1 | 1/1 | 0 | 0 | 1 | 0 | 1 | sha256-cefcabf54971fbf2b6cff77737a13f8199666de28824adb486e160f492d33de9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b53a4d04b60344c4edcaefa8c71e165c66981e9f5d3ff18c113dc25dfbde68b6 |
| learned_route | 1 | 1 | 1/1 | 1 | 0 | 1 | 0 | 2 | sha256-c900cb31ba9d6617cf8f59c8bb0eaa91e288c349f6e2540fec034ab019f4015c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-e2940a42 | sha256-37eb7e860d255ef599de6963e9ad8230a3aa436ce85e2333b6003e1cab53cb35 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-e2940a42 | sha256-d4fd9f41fa562232075dd7d46691df0d40da7d261d3e2fb24e55a0673f78a33b |
| learned_route | turn-1 | 100 | yes | 1/1 | yes | no | pack-31aef2e1 | sha256-fd9b453897a5c32cb7c7527d8a3dfb7e87889d1d78b4bae019fcc18fe2b99917 |
