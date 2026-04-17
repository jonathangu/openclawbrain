# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-34bdf8f02ec779363fa4c8a951850c78f5f147002a61b16879fb9ea405e3f18a`
- fixture hash: `sha256-fcd33f2f91aff8f0b5df7411e8b655364e067521c4da8a9896b3ab460088a1d2`
- score hash: `sha256-fac9826a059242fd92815f5c9732b46f6873ff189bed435baefc6145b06c5891`
- bundle hash: `sha256-8716251c21f854d0d1781385e9e8e1003fe198b3c18030e641079be00e7f0268`

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
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-7a1a5b220c2bae02c0cd1afd83cd084b70f087d32b5037b69001d33fe8106ce2 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-c9d1d13a757b60711e4707b6ca464c291f1cc4b3124d057f0fb129c318becad1 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-4cabcbcde7f6bfa19a0910d0a28d08b05fd7f0bb82b3b369c50655703a01459d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-333baf44 | sha256-2d28135e2c402b21a2c1b1fc46c9301d099d07cf2e3de8277ee66c4e16c9e943 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-333baf44 | sha256-52a4257a9a1482057293a1834024b4a1c15e1fb5cc1de2306c977c3d541fd8ee |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-ddf6528d | sha256-9e3c6a49aa4bf857c74f62b5151287cb646cff1b68fcfc9a9d6e5fa8dd47dfe1 |
