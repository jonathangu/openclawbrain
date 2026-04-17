# Recorded Session Replay Proof Bundle

- trace id: `live-main-dd9238f7-bfae-4ab9-9640-9e63a04c89b7-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5f45e438e0f99f9d56b4e0ce3ef341383c4f2368651efec5583a2c7447c8a5e0`
- fixture hash: `sha256-24e221e1cec238f614a332fafbde124000574c7f4eca983f394d512d73646f16`
- score hash: `sha256-e8a34b312b175ce244c6d19080e54585b29702148e9bb36f74947e31bfee1a51`
- bundle hash: `sha256-29bb679f705ff2574da8986b51585c44b25381be3158cb467d5b4b6258914ae8`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5624a20b92c6a5c4c5d269dbfed46d621fb3009b7407cdb61d3d2abad216a892 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a36061df91ded2fcf4f00f0d5249c70953766051e7ed67af5e64316b6ceaf16d |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-f2a831ae694497d86e49f6c1bf12c3fe459fd03478d872acf56641267d3631e2 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-f313353bcc740d6b458a8afc8dc04b837547ba73b744eb99b325f23a6f149b78 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-206e29a3 | sha256-4265cfe8727fdf8d544fdc63379d5f5de4401be8a8e9bc9f80d68941b1a74799 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-206e29a3 | sha256-a583dbb26b4d7541058b9a0a8a48f7c98b3777215be6b514a81fb4521b23ad28 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-0df7aff2 | sha256-bf5f62288cc2605357f814df130fe391a396ef592abc57f5aaa0f97dce2cbdb4 |
