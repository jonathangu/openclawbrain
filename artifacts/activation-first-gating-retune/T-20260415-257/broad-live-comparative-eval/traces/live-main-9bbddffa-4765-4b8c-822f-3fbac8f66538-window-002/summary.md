# Recorded Session Replay Proof Bundle

- trace id: `live-main-9bbddffa-4765-4b8c-822f-3fbac8f66538-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d82a353da8dcdcb26be266a0d33a583d117e1f2075d582930b7ead32e3d715fb`
- fixture hash: `sha256-3e1329cac030395635745dbbbafec0f460454aaccbb63b26f155aed0ae65e7c6`
- score hash: `sha256-95cd87b527071d6c1fdcdcac3e22cda163e847a3e133e0048a92258120f828f2`
- bundle hash: `sha256-294b61803bd7908a781bb952cdb4e1268d1bc2d75e8fe1f9356b0cd860b59630`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-38cc6030cda1cec211011cfcfcb3fe3c0763917e1cb1cee36ef6155b409ff4f0 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-2ecd1f6fdb8b849f9b4b8c99647e35e01d6b96fc5d7b272cc6725488d5242269 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-91f43d348e4dc57136a814fe15310e40ba78578f032bce10574c2f19743ef9bd |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-7b08397f18f74a53fe1ad20941fc924489d519d2238e7be7a6c2eea17cee5469 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b1fb0cce | sha256-383dd1f4d7d73c7c897ef86f8902aba8c2f6ac497e56afa0cd538da8c09fd9d4 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-b1fb0cce | sha256-878c30e1a8069ee6b4b6e967700d62c90d905dbe3e5a77c62a5e5cd48ed45be3 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5f29c179 | sha256-d74c1b5ace876e69d68e462635514e754a7ac383df1c8586e70413920dc243a4 |
