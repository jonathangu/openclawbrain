# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-bfcdb554e6c3bfe187f4c905f92a9b282d7821367cef535897c2815e123fe75d`
- fixture hash: `sha256-3907274214cdd60210f9dcb9d9b0e865d090d5365a59db918b98e4ad4849f4e5`
- score hash: `sha256-be99e2827657161cdd852969b66232e33b0c8e7a40a2e1c3262c38017656dbd8`
- bundle hash: `sha256-b19411795fa8a892589f8482547f800a0713ec1366bd48c21c1c1dca45cee9ba`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-18ae191771eabc01fba0eef9c0e7f277194aa1ae188e2e94481f667ee00cc41c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b2e1f56b1a725cfead1a3a1bf9102a5f860b0ed03848dff202833c67fcab3a4e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-02583f2863418519b85f54dbf08fddf600b713b46c092a946dcdd65363bb8bf6 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-e11524ca7cc41bf04eaa2a0618f533c819e27b955758b1b569b8b12d47bc4a92 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f1a91c99 | sha256-333559afcf49b4a014de587dc107cfaf7403a268e8b5294e5ca58a011143d120 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-f1a91c99 | sha256-f0965932f4b3951486512ad2167efde620337adbf308d441f3c86fc0f2c60437 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-f1a91c99 | sha256-333559afcf49b4a014de587dc107cfaf7403a268e8b5294e5ca58a011143d120 |
