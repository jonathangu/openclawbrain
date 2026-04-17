# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-014`
- winner mode: `graph_prior_only`
- trace hash: `sha256-82b94292f904129190996d09645352442519cd34f4a6fe4ddc3d8ccfdc15ed4f`
- fixture hash: `sha256-2b7971a9291be722d620678727dc2afe570e5b9dc9a97d0983cbb8375a8b4f0f`
- score hash: `sha256-ca96405f38f73befa34a603c3fae5276d2c61702ee773a38af07833e6b63249f`
- bundle hash: `sha256-f7f6625bc5164e965335f5297d4fb4653ca3c761efdf718e2214c46cd0d53a1c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2126389181abd46124f339c97d016b2e80dbdd1c3f4a30cb14b5104924e09f3e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-1b6fa9446a05307b9865d2366ebaedad062cb91c94af285a6b6dcb43801fc90b |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-6c45d65847b1695758e0b3554a6e77e451a70e0157ba0c8c6f4e1d4043409404 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-df6b7a878df835ec270ea977a6d1a0e590b14afaac1cffdf5d800c2b700106a5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1406fec3 | sha256-b434882b0a6a986468dbea459b38733ee925e0bfdbe646157e4c8e508df9e545 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-1406fec3 | sha256-25661889470c0a604e03a11f2f711260fc8f3ba544ed404332fa37ac9bdd38c8 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-a8da08a0 | sha256-31d77fc97a5fe454293fb334f78e0e7baf3aa731389b58559f5059e276495802 |
