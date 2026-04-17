# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-007`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f65f7ec4c1006917225f6f3df2297434078972719c9016d9a4a28c343601c090`
- fixture hash: `sha256-0846d04b26eef0a1a7c06190a5a1fd4f54e0a1ec3fcf3231ae0df203565132b6`
- score hash: `sha256-9e404c10ffe47c61df350145d707dda3f60327cdfa2ed6addb4334fb96e42e03`
- bundle hash: `sha256-31042189ae1b7fafac4c2e2aca99db79f1985735dfd5fbe476ebc1c764d73b30`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-42f48b48c6c450f0664e256db3a267d908035a318a1c9a74a979a0b9949d1634 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-d26ab76a093404ecab1f4902e9a2e35de09c882f32f8cc3f4c42660f95b69683 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-73aa230caeb33fbf88015200a922b22a4db463e5cb521d01b5bc2c268cdd0fec |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-690b15f0f8895f527d45bb9286c0748ac2cf4c67100327534aad2221fa7a70f4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-70e62622 | sha256-7b990fd7b1dd9a6713eb4ed17b111196489c0814d2a2ce3c9e1c2ed5147bf0d5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-70e62622 | sha256-f5791f00ef45efb34e3424086bc41194b1b57ac30576d5538c58ce81e2b376a8 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-293ae087 | sha256-422b05d99f405fda6174626b069fb0886c5058ce888770949de543ed5d1d503f |
