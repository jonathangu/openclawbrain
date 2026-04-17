# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-8d942e32-c1fa-4af4-932d-fc1e8cb76bbc-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-34bdf8f02ec779363fa4c8a951850c78f5f147002a61b16879fb9ea405e3f18a`
- fixture hash: `sha256-fcd33f2f91aff8f0b5df7411e8b655364e067521c4da8a9896b3ab460088a1d2`
- score hash: `sha256-2ff43d0711ae8e2672c1534d00ba325710aceeea9836d0fab65f01549be7e103`
- bundle hash: `sha256-44d3774637b465b75810d402069580aeab6ae69e218af27b738afcad68058243`

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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-6e9909213c7dc1efdc972df68cc95206f59685fc2daf810999ea603cc01e47d9 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-a1e1373adfd87dd3b36273a052c866bfcb64f6eacbee788ad673bdcf1fe07de6 |
| learned_route | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 2 | sha256-d22ea2109f04f314529f9cf19055cc5a43e0d14030ee2c75b7ec6e810816c235 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-8d691c06 | sha256-77d56f9525e1b3e6afd4f687cf3bf4620cc4596ebb2601ae4eef36bdde716eed |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-8d691c06 | sha256-19f6181abd3a5c9a2f991d9c82ab6e45264ad8e446a5d4e78936d469cbb24c11 |
| learned_route | turn-1 | 60 | yes | 1/3 | no | no | pack-3823bf4f | sha256-69cb0b934311c8d277616c13d7b45eba1d78789c771b4bdf85b02c8b705ceacc |
