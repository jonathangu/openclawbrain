# Recorded Session Replay Proof Bundle

- trace id: `trace-retrieval-proof-hashes`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0459c0fc86792c696bf9cc08f781d4cda302256c62e036040b83468b7f5434aa`
- fixture hash: `sha256-037ab45cee260e72054bca957f17ffcf09f42052b826c8be6335ab9bf5a8cb0a`
- score hash: `sha256-0dd1aebace8e7519ab083e55aec8f6c21dc93108dd95314bb3b3545e42108f12`
- bundle hash: `sha256-9e8e59fd0c8d5dea2b3fa97e22eb2147e084a9e9c0b8fe99d148f063d6356799`

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
- phrase hits: 15/20
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
| no_brain | 2 | 0 | 0/5 | 0 | 0 | 2 | 1 | 0 | sha256-92b6e3cbb25607cbcf4bbb960bff7a027b61da7359fed6908c38ec8cdeaca5c0 |
| vector_only | 2 | 2 | 5/5 | 0 | 0 | 2 | 1 | 0 | sha256-b358e9f86119a0b28c9d1a927db583ff3b74da094c2cc2df652c309c45cd4441 |
| graph_prior_only | 2 | 2 | 5/5 | 0 | 0 | 2 | 1 | 0 | sha256-1286390e33d6b579a7ce35bce914eb7364fb7f16adf70fbd9a163a58fbd06482 |
| learned_route | 2 | 2 | 5/5 | 1 | 1 | 2 | 1 | 0 | sha256-a7947c5efe444c9ff6d355b3f62d98886f57b16676e8657bd6286bbae453584f |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | proof-hashes-turn-1 | 0 | no | 0/2 | no | no | none | none |
| no_brain | proof-hashes-turn-2 | 0 | no | 0/3 | no | no | none | none |
| vector_only | proof-hashes-turn-1 | 100 | yes | 2/2 | no | no | pack-0e280e72 | sha256-cfeff7fb40cb75760660d4bbf0532273bff2580d6e11a55af79b61932be39ee9 |
| vector_only | proof-hashes-turn-2 | 100 | yes | 3/3 | no | no | pack-0e280e72 | sha256-cfeff7fb40cb75760660d4bbf0532273bff2580d6e11a55af79b61932be39ee9 |
| graph_prior_only | proof-hashes-turn-1 | 100 | yes | 2/2 | no | no | pack-0e280e72 | sha256-cfeff7fb40cb75760660d4bbf0532273bff2580d6e11a55af79b61932be39ee9 |
| graph_prior_only | proof-hashes-turn-2 | 100 | yes | 3/3 | no | no | pack-0e280e72 | sha256-cfeff7fb40cb75760660d4bbf0532273bff2580d6e11a55af79b61932be39ee9 |
| learned_route | proof-hashes-turn-1 | 100 | yes | 2/2 | no | yes | pack-0e280e72 | sha256-cfeff7fb40cb75760660d4bbf0532273bff2580d6e11a55af79b61932be39ee9 |
| learned_route | proof-hashes-turn-2 | 100 | yes | 3/3 | yes | no | pack-faee9a58 | sha256-93668334f942bfe19d761890b21abd360602374d3014a9aa486d63753745a31b |
