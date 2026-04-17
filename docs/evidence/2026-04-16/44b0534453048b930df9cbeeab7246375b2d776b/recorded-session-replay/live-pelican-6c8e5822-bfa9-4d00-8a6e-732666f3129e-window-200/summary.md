# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-200`
- winner mode: `graph_prior_only`
- trace hash: `sha256-438b689c90e1516f117c130a44f955ebe5121f19131ef3c8af4f3b72e782a392`
- fixture hash: `sha256-fef64d4e61173927de1b8c7e42759f7ee5918ab3e67738573626a046f39d5b5e`
- score hash: `sha256-29af38ee8ab83a3307a9bba0b383c709c4ea739aa77ebdaf7748ecdf74903947`
- bundle hash: `sha256-ec77949ec672f373af55143a28285633b5b6a9db19e8ac76644397751487a5ea`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f3e6c6ba4308832d244620436e1eb71e4969051bd02e8a257e4c9a12dea8653e |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ec27fafd8d165e3eacfdaad479f8e9702c818807cb4e3a55a956a3a6013988a0 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d7835ff0bdd1c3b5e306198249f57c5136146c2fff5ac7989ddbc8e3edf39825 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-576f3284027e02d74c1d8a89cf1c4e96893bcbe13c1e073174f3062e351caf5c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d81ae853 | sha256-ae360f239a10a35556907c8a702c489280c966ca7843ae8535c68ce7a06f03a0 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-d81ae853 | sha256-3db715b1d7b090f35c6d9e57a38eb0a4659cba98b15a51e15330cc55917da1e8 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-42386e9a | sha256-74aea39953792453bac7851d9d9a738d40f9968b07cd372a524c889271622dc0 |
