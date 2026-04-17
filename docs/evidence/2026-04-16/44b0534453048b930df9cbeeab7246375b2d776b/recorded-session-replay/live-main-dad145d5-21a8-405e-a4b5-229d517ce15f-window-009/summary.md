# Recorded Session Replay Proof Bundle

- trace id: `live-main-dad145d5-21a8-405e-a4b5-229d517ce15f-window-009`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f0f24d3812e038d9d2b67d9309de9db96cd24c2faefb0b5dd93caf569b3c1d1f`
- fixture hash: `sha256-6b6c634b067ee2b84c6981ae8fc0d6c41efb6194e1723d88dd7d0087036cd1ac`
- score hash: `sha256-ad3a0602bdb94ce690b05e5c101eb6d3e13e920116e4ef7172798ada3fcf8ea9`
- bundle hash: `sha256-232cc9bef0004dac8afe5e3c4b0ae2cd13f6234b6e693cc74222bbb44c334b8f`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a0ea26975a8365e08832501930a2890706222216fe363c833adbd0065a774a3f |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-b81e50ccaaf4850587eeeab1b5f3e4e5114b78a1557b067cb232df832d5ab671 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-46220c6332f88176fb8a1302d897be2f6b8011fb17ab490985fe73d727051131 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-665e3eb48031cb3b8c38b7c665a044e222d728e5395048aaff4f19673363991f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-733ac76a | sha256-bcdd66f825543028e144d754705e96501155930d1d95e656165bc9b399725d0e |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-733ac76a | sha256-4e195e143d83e26037a4af1f9b019924edc66d4b9a6d7f23b4bebd4a6d7dfb10 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-73cea305 | sha256-f3c38c39eff0e687ba01c78b61da10155776b28d470101daaefc7dd58073538f |
