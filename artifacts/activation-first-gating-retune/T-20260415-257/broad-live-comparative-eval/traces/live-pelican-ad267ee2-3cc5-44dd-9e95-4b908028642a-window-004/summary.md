# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1382603626a2aff7d92c871c45318305722032a646fc01502f912f8472d0ed38`
- fixture hash: `sha256-ee8d3f8c272648220db4d9e69e984cdcf85084bd085927ab6802512d77922517`
- score hash: `sha256-f29c02ba576d60130ff3f5e27b5bb1540fde4209d7082e34a2a971421940b059`
- bundle hash: `sha256-74f2b2ccc2d5ed4689f451efe8d950f653e94bdded4bfb0b339279d8fafe7d41`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-394d692f2aa5412e9da10dfc0baf182beb2043f517fb99b07451a27af9201624 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5174cc42473244779d3f1ed5719d9694c60598fbc0c9bf97f55b5d94efde7152 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-83163b6e8dfb21bf6c05ccdb0bb0e1ff5f938da6a1563253a24d545bebf4b50e |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-28bebc789cf411c65fc565aa32a91ee638f4606ddc2d105b160c8052b19dba66 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a02f6c6a | sha256-005d8ec70868ce47c6f3e59ec1caa36e15fa7ccc0bbf6a317b81dbb05beddcc1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-a02f6c6a | sha256-3bb79536c93c4ce1a24bc1f3586617be4bfcd798d82b3844c46bafa1933bce0e |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-955332f5 | sha256-35b47dd06b63933dd19168b3d7a7b910cfaaa8db2ce8c9ac5cf124a6a3670c1f |
