# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-155`
- winner mode: `graph_prior_only`
- trace hash: `sha256-54c6a7f75aa98b64fa06de64444db8f288aa41bfaf9731cc070d54f577be960a`
- fixture hash: `sha256-899705aeb2321d03b6a0aee78d7cfb19ca0d976080db3e6a3f83db60267852fd`
- score hash: `sha256-6c402098b8968cf95ffcdabfebc858a0c18e8cd249fc99077d53aba4097992fc`
- bundle hash: `sha256-936c45fe6ea4dbdcdec6bc7af1215d5b43e8e4c43a213e861977fa07e5f1c82c`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-ee198590c5a4a8c84e2f8fe36017d040fb15fc92428b4d0396417de634b42329 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f9538ce46f04fe0c10d3a58607a33a08722f92c4594e2c9f873c49d3bc50d83e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-710517314aa96b0b952045c14d0ad04784c1f611d54b8aff3fd0da219a39cbff |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-621c25e42a7924352230c072b102b778cd545f93f98487dd2dd522637209f977 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-288badce | sha256-e59def6ce765e43abd9f7f82e5905502fc1236953f23282d89ae7619e11bb9fc |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-288badce | sha256-3cf11aa1a8e6ce2dcb6dc22386157a1868615ca1ec20c27fdc0df80ecc544fae |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-ae21e833 | sha256-a158b18aff68ed4c051bfe21d85f00a4be844e02ded8fff656be21aa9a15269e |
