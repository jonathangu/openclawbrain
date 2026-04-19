# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-184`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d479c91d117044f49ae49da499694fdbe9a9bce3b101e2f906d0092b46536940`
- fixture hash: `sha256-afb53fed27fe0fd6a6ad4e067cb4e140573e8cbd954bfddd658b0c3c6c424a0e`
- score hash: `sha256-71eb300b2e619641127e192177a6c48dc357b1b6e496a16efa6e3bbf6523e673`
- bundle hash: `sha256-211614674fb273651fbe904e39b4d51c8ad6bcb22867e0e912df0fa8fff6fa10`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 80 |
| 2 | learned_route | 80 |
| 3 | vector_only | 80 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 6/12
- phrase hit rate: 0.5

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.666667 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.666667 | 1 | 1 |
| learned_route | 1 | 1 | 0.666667 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2f4b660a3e5d5f4a920994b92d0eae72726c74b613f3139fcacbac22692626d |
| vector_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-ab6439787b21fdd9419581c4503bb09dc5158b9793ffeb43256832ba4cb0ef00 |
| graph_prior_only | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 1 | sha256-247e87a28123016aff46b6a625e3bcf33398a1474b5b4b730c1e71b2bf3fcb7a |
| learned_route | 1 | 1 | 2/3 | 1 | 0 | 1 | 0 | 2 | sha256-2a16ea874031a53e978fe9bb13d7b15542f4816d3a5dd632a7f0d7bf73e50900 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-0125025e | sha256-d374a11f29cb1fc454e18944bf2e4b69ebe8521be2d77ed3ae1a1bcf4fff1fde |
| graph_prior_only | turn-1 | 80 | yes | 2/3 | yes | no | pack-0125025e | sha256-409465d17f21edf2c865e1fb87755072276e800024b2205d3fb7eed847029516 |
| learned_route | turn-1 | 80 | yes | 2/3 | yes | no | pack-0125025e | sha256-d374a11f29cb1fc454e18944bf2e4b69ebe8521be2d77ed3ae1a1bcf4fff1fde |
