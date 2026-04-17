# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4e9af4bb-5f20-4cfd-85d4-a00bb3d64878-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e3f86fd026217c7d6458e87e96268ca58f7633ecf498ef1f8793a6a7617c13f8`
- fixture hash: `sha256-c25bf3a6bec00b35ab13366d1787d21cc5e0fb28011aa90689176fbd43238498`
- score hash: `sha256-c12d2ead7625fbd26046f3de229b87f24717bb12f7c59696829d19003eaa0fe5`
- bundle hash: `sha256-ad71f1044ad07e9e916ef539e573688b201dd355ca9401f0d020257cbab17f63`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1e36e95d3b902dbb1cba84b7196a751790c689dc2e631e7340724bc6d85c3a59 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-377018360932c0c005e843249d8365585fa5a12372a7eb76fe4c107d3cafd669 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-67e0a7c88f0f387f73e5b9d9e6a2748eddbc918df2767b98755fdeee81795203 |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-cae602ed2171935d1ac4e965700e757f3874e64b08e5033db674d5a9af9cc189 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f894deb1 | sha256-a40d6893efd83d6f2e5dfd279420451e747ab5ce1a393fe7beebe9503a8c35ab |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-f894deb1 | sha256-cb2d898f197e3581067e8ef9b53e268e144c1bb76c4453da917ce48378902d69 |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-75450256 | sha256-97a25ec9681e3bdccc1e17c548856a95e015fbf055d7383d336345e10f73275c |
