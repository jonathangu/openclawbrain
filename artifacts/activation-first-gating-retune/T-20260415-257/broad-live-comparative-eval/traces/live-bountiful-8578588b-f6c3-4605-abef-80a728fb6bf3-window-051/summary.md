# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051`
- winner mode: `graph_prior_only`
- trace hash: `sha256-87081ed5c11e35636d4070ee89bd588bb6f63117995b8bb493b8643be003ec58`
- fixture hash: `sha256-43945ff6ddb3d96c1170069d079615a310c124eadab260a82b171a25d542a6d7`
- score hash: `sha256-04d85c3158e9bd378ab4f901bff8a28005175d5069a33607c1b5825f8c06fcfa`
- bundle hash: `sha256-76900bf743f813349108dc7e5d5fa415c0a576c578e2c5c95f09140c692a6b5b`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-427320990f4549fdf09c9b73d9eebc938af8aaf238ccccbd97c3c1df3afef6b8 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cb7bff96ffb7e98b9a87eef8aa53a9262fbc86faa7713993cb4848b231a05f93 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d18cdc6bbf3d465d944eb6e3267cfd990d8adf2bd5a5356c70455b90afbe6183 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-7e3ca54b50fd30b7c5555cfc48e27ee27c5b8eedcddea673d65f277ec488ab44 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b0916f39 | sha256-40a08bbe0b800430c127019094122dfaa2819d578531e97c2aa8e552a6c7a09f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b0916f39 | sha256-dc5114b1cad011969d1940b70259c7e21135d4328ed6363e4d1c877214515a13 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-87708f44 | sha256-8c2a8b7a7858ed8a62127b68078fb1d2068723bc0bf6d1e1ca658cabc781f001 |
