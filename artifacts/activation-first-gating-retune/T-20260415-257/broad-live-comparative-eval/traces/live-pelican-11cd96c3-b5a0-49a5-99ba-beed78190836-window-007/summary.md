# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d958774a8fc5556f6b626cb2afd5141be38b390f01c3f1c481f5689e5c67765c`
- fixture hash: `sha256-bf711d8c588faf57d4df6088b8652fb030ca7a163bb118e31c3e2f2768cad0f2`
- score hash: `sha256-2779305a6c572280ced7e2929a814406c23a309e4abe4ec754633f3e0316e556`
- bundle hash: `sha256-a02e6f94d421a3be19d4c7cbeeb52a5d172b49bf3fd76e8cd1428f7bee85a016`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7f7ca30e9c8433554610f300b068b172fcd1c7c716d277545f4d5940081fb358 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-65b91086c3769def1119411d1d2d71fde3731062b85e83c73cc64826680217d3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-221ad85647e2527ba134e0531b020a5076785ccf6759d3de28595e8ed61cc0e5 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-022e6cb14e63144acc1828298da510db6e651b1dac555c02e2df667707a44686 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-93182107 | sha256-15b1067bd859fa0fb80c6268551ffa6b07c5caf711dbe27358b92ba0138b1f7b |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-93182107 | sha256-6983818c2405e9ab0046347379558f6e07a00d5167070c83b2212c1c4276637d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-3d75b666 | sha256-ca6f2deea3d5436c0a3afbf676232d88a6eab7950f5c431d94e18155ea1e8636 |
