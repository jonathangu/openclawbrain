# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-ad267ee2-3cc5-44dd-9e95-4b908028642a-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1382603626a2aff7d92c871c45318305722032a646fc01502f912f8472d0ed38`
- fixture hash: `sha256-ee8d3f8c272648220db4d9e69e984cdcf85084bd085927ab6802512d77922517`
- score hash: `sha256-f259a102c49b0e0491029e4fa42dc7b7637162c3d45bdf85ed23cfdef94b5287`
- bundle hash: `sha256-3c6739495a9035c1a2d4c210c35851f486a864384e61ce6b92e7f8b96656a5d3`

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
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-0c654319a96bcc1efd68fc1f811001964adfe5ab2aed334071b8b149784feb56 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d100be08665a7880db5769ea65383c65ad802448e164b69e6f6b237b42d95744 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-591db7c44a6eb9604fc560fa238fab25e1ddcbc312b340ad16416b73b41e454c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-31d1eae8 | sha256-fccc5e2b48a819056abf6f59df7e44fd7fd6cfcea0e8c0735e8f6ba1ba0363da |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-31d1eae8 | sha256-2eae4d5876efdbc1b0236d5ddd2a9505ae785054844743061db639ef086ca69d |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-26f5b173 | sha256-80309862b88aaf42e04f9a6d815dd38fd48b6e15a3f5b261ad177ebff3c871b5 |
