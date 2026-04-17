# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-2362908b-54fe-4301-aaaa-003f211ba89c-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f69683abc74146be49e8afbd73d2f629322351b8f1ff326bedad7089f23b35cc`
- fixture hash: `sha256-78ae89352ee0e2620fdc9e4b5d6b74ee70bb4cf28775ccac9315ef7f4b6b2525`
- score hash: `sha256-49165906215e53dd02bad0149681a5f27503a0093942a263470626c569d92fc3`
- bundle hash: `sha256-65e4f2347330acb5fffd03ea46b599aeab87832f2e5c774bbd4dec72a1ed148b`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-81a98d75515ca1c6519d32d4f8b5120338f9765022c93b90e0504e9561ef38af |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f110f44daa70c7091441551cc2d97e536c5f10c726a3ed336136e7f9d4af0c03 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-7963d12dc4d6f041a1216c6333f3d8e758c29e0a869b260a598f9525e4ed3085 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-e93c53b611d3806d4fcad65610a97af110ac6d4f5e5d8aa5d095c49869eeffad |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9177f55c | sha256-c908d29872c500a52899c8346da0d541f85b62de6a54f2ed0c53f4d5ccbdba20 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-9177f55c | sha256-40f65330c37e9cb76b963573af2be3441deac0c7150d460564550508fff50c95 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-c7cb2b49 | sha256-753647593b15c105234ad0dd3307585b46e0ee28583bcced9ead4b067c123d1f |
