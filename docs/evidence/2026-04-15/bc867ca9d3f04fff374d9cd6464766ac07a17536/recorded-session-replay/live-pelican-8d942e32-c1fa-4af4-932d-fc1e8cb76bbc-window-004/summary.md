# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-34bdf8f02ec779363fa4c8a951850c78f5f147002a61b16879fb9ea405e3f18a`
- fixture hash: `sha256-fcd33f2f91aff8f0b5df7411e8b655364e067521c4da8a9896b3ab460088a1d2`
- score hash: `sha256-a189ef06f0d4ef797334b2450d92c5610e82574e5f91330e506cf51ab05440be`
- bundle hash: `sha256-f5b0beea0e5601e80b20f55ff0fd0f7a683af31afbc528d9aa2b680d45775a85`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
| learned_route | 1 | 1 | 0.333333 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-59aa7e4d7b28a1f5c691ae41334f8392171bd8edbe01742338c29f7ed9b2609b |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-8a00fe2307ee8cdc0833e10d4d838a50e7b4512d855bb5905678e8bf73f4fb8c |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-5a2fbbda0803020959e9748ceced227f2ce035da19d374e61bcaf09e9f825a2b |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-9c102f35f1c078a3d8299b0bf18ba5d8d07c7f84aa9dce53e6c78bcab91d1bb6 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1618d67c | sha256-a74416d4d9fd96bf898944386f7e39498024c3a95872a04a5b8c01ccf98b829c |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-1618d67c | sha256-ff0270afdf94b2de451784a1ef51b41bb23daef223efb7723559369df3c22d8e |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-1618d67c | sha256-a74416d4d9fd96bf898944386f7e39498024c3a95872a04a5b8c01ccf98b829c |
