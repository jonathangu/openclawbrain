# Recorded Session Replay Proof Bundle

- trace id: `trace-comparative-replay`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b95f895c64041d1808d9fe91f67bd8bd003d088a62ca59cf1e440938a201e26f`
- fixture hash: `sha256-279a2c9838f639bc9a2c4c0580126d227be88558a0c39fd66ed2315cce401582`
- score hash: `sha256-58e03274c19149e475c33076a5a64818ff5ebd8f702f6f8b593c3515832ad225`
- bundle hash: `sha256-299630c6c6f4e4ccd8d4f97db94f2948a4545d02906740e5194903a027e919c2`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 6/8
- compile ok rate: 0.75
- phrase hits: 6/8
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 2 | 0 | 0 | 0 | 1 |
| vector_only | 2 | 1 | 1 | 0 | 1 |
| graph_prior_only | 2 | 1 | 1 | 0 | 1 |
| learned_route | 2 | 1 | 1 | 0.5 | 1 |

## Hardening Snapshot
- compile failures: 2/8
- compile failure rate: 0.25
- warnings: 0
- promotions: 1

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 0 | 2 | 0 | 2 | 2 |
| vector_only | 0 | 0 | 0 | 2 | 2 |
| graph_prior_only | 0 | 0 | 0 | 2 | 2 |
| learned_route | 0 | 0 | 1 | 2 | 2 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 2 | 0 | 0/2 | 0 | 0 | 2 | 2 | 0 | sha256-bf3e5467de37132a2e81d2ad5eb47a8c3fbe8fdd240efb52eab44760e5cd2955 |
| vector_only | 2 | 2 | 2/2 | 0 | 0 | 2 | 2 | 0 | sha256-a47730a1714914bb3f9366f721b8846e84c304eb1e578681a941084cd8c6aecc |
| graph_prior_only | 2 | 2 | 2/2 | 0 | 0 | 2 | 2 | 0 | sha256-a3ac5a325a4091a97127578b77ca154f068462ba59a1d97a6d621bcdc928a435 |
| learned_route | 2 | 2 | 2/2 | 1 | 1 | 2 | 2 | 0 | sha256-3605fbec39516048e755ed0df738dc5e95971086868e7d862ffd21d6f281c9d5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| no_brain | turn-2 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 1/1 | no | no | pack-d2252553 | sha256-f1b76b33bafb7f58e23296e272292ff19b86d0ca91804ce6628b10ccc06929b7 |
| vector_only | turn-2 | 100 | yes | 1/1 | no | no | pack-d2252553 | sha256-f1b76b33bafb7f58e23296e272292ff19b86d0ca91804ce6628b10ccc06929b7 |
| graph_prior_only | turn-1 | 100 | yes | 1/1 | no | no | pack-d2252553 | sha256-f1b76b33bafb7f58e23296e272292ff19b86d0ca91804ce6628b10ccc06929b7 |
| graph_prior_only | turn-2 | 100 | yes | 1/1 | no | no | pack-d2252553 | sha256-f1b76b33bafb7f58e23296e272292ff19b86d0ca91804ce6628b10ccc06929b7 |
| learned_route | turn-1 | 100 | yes | 1/1 | no | yes | pack-d2252553 | sha256-f1b76b33bafb7f58e23296e272292ff19b86d0ca91804ce6628b10ccc06929b7 |
| learned_route | turn-2 | 100 | yes | 1/1 | yes | no | pack-b06a5e31 | sha256-ff1c14d8f7515397d91a7564580950f43f42960deb4963c6638ef1617530c287 |
