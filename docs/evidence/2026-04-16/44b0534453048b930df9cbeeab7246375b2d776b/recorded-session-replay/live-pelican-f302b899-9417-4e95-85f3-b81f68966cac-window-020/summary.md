# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-f302b899-9417-4e95-85f3-b81f68966cac-window-020`
- winner mode: `graph_prior_only`
- trace hash: `sha256-020e5fa0ec60c9180b8ca12d4a8cde03c3eaf93efdc6e1249456178218366170`
- fixture hash: `sha256-2fab851b07744bef46921e5dde6e3c44cc707f0e47e7a2b971ff5ea69c88de53`
- score hash: `sha256-96bf997541a1f1b6ef29d84e61fa49be068ce99524a41c688fe0dea649448898`
- bundle hash: `sha256-b1dde2292399d1e2548204ce3271a04e1169003a12e80420774dbe89554e64f3`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d84e9ddc31e34697064a9e60de43374da82ef3d65551bc6676137ee0e90f5d63 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-bc1b61e5047a4fb1217f876c4518fbd7d337f20a6b64e4fa44db6c8e98e97657 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-e8329b0a87e6b21df68f64bc94f6a1ee8d44098176f04465b802d435ff34ec17 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-96f5026772b5ea3261f01d4f1abfdf8d12a039c9f2c8127515eef1346e08500d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4b0a51b | sha256-a676f382b961e9a06cef21fb4863a8ac8f1e59be008da8db32db7c9f7665d50a |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d4b0a51b | sha256-130e5229a56458ce5a28362212fe5e374382f1749d530f6d17f4bfed260e5069 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-6ff259ec | sha256-b485b02df74c60dcef43d62825f52f323a0ae83c89f72f275cf362d9935dece4 |
