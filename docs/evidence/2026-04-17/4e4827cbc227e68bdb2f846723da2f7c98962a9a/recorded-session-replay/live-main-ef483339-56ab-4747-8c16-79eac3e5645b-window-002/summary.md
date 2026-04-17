# Recorded Session Replay Proof Bundle

- trace id: `live-main-ef483339-56ab-4747-8c16-79eac3e5645b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-5348a011d171022e0b0662292622dd790b7dccb6110063ebe79c7f32c96cfe4b`
- fixture hash: `sha256-076839ee1f2768f5fb0e1a395f80dc28e7868b4aab96a489d4fbcd347a8fc395`
- score hash: `sha256-2b12ff4f61b149059f2a897e5821e6a3bb1b3e472ac61e2cf27c8cf97c71b6d4`
- bundle hash: `sha256-e66403b17576f580f908636305add8d723852592d0da94c359ca11eaa2139137`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-81e33f098e21a8124ae6ec9568c1d72b0f83fe94e5b59e948eed0392a9dc9438 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-fcdc02cc040e38da5c0a7762c42b623c4fc8d0c0778a29f177a8b065479b68cd |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-dac2f93c1fcdd0d2009f2a78b42dc316aaf0e94089994e59fdc67c5f2cac14c7 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-38d2a38e3f4f20d38e473715f8f254a69764467c7ba56486a44d03468ed4fbf7 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d1441006 | sha256-a4ddce77f16b4af5b536cd0b57aa3bf23fd8accc9a5d1fb010d9fb321d49f9fa |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-d1441006 | sha256-25c49fc1c5061322b2aed167367b3fb1adbcbc178f27da620d522e9be1759693 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-10049ca7 | sha256-15499c09e03a90214068aa719b27db6a452d6ecc5ad574702ac7677df78c6707 |
