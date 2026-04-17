# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-034`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b1cc11a021b01ca8aba99240d6af1922e93eb57eb02a859f34b6e95e9d51abb2`
- fixture hash: `sha256-25597cd73d9bd0f0d086440e05e70594b904a1395b870c35b683a0a720d202a1`
- score hash: `sha256-fed49c26a94e4cf07428c164f49c1a5e0e65ecd09931d59c89609f425fecd7cc`
- bundle hash: `sha256-ee8b7a6beeaa5ce67d4dbcaee903b92dfff28a6b2e49f9fe38029eef1a03f47b`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-8e0e779f7703426d9dc6e56462e7c187c6dea02bfd7bb266c409658015b58695 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dc34c82b688e92df84f420d49b43d82071175256557ec64068e4615f7b65df1e |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-6102cb40edff01c4b63d5e64bd5feda152bdf114525989c219bd794c2567d35f |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-f0c2bc3687b495108f8f4670a06b00f9ea866e416c1940bd2e5d251ccd6ab429 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0bda32cd | sha256-9b8e99023ebf2f6234cc37beddb9fde4c34541a9b783d3fd0b185cdf98d48cb4 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-0bda32cd | sha256-6dc998efcd4a68e5871f534773cf4e8274382f3321ae6d858464978bbb33809e |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-d6e499cc | sha256-b7a0c086dcab159c5095a5a4973f456d688ae85ad6266fbeab6e542f74bef3ae |
