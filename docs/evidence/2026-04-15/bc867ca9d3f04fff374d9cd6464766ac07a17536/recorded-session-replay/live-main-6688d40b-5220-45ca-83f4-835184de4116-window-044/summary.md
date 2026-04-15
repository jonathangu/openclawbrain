# Recorded Session Replay Proof Bundle

- trace id: `live-main-6688d40b-5220-45ca-83f4-835184de4116-window-044`
- winner mode: `graph_prior_only`
- trace hash: `sha256-69ce0c4e11baa36853be20e1ca688e734c8855423d37366857eb233deb6e9df0`
- fixture hash: `sha256-c3a333635db8e86be19e8bf48de8cbd13aa6939830c506cedd85267cb0e9f51f`
- score hash: `sha256-3310d72e30950da03381517b33b6bcf09e10e1f5ced4cc12000179c382e8bf3d`
- bundle hash: `sha256-91e96c1997fd4282a22b12b32c7382a93da3adb7a996120c86b4bc46662f46e2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-653724b1c50980255f17a34150c96cf9693658619075d0cdd8b7b4b447cb2cb6 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-002ce0a5cd1843124fd43794208685369c743256bc8f288aa78593790a25e16e |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-3c849944a88aa702d315da5c51b5f1d10abef30d100c7b04d40cde57289bd990 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-cbd8cf8122c32018d48cc41bf029889cfe4494117ba91d23531f7e561f00011a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5b254b59 | sha256-5d60982e8274fe72179ce26aa2294c01ec4428badc627db3a3951e5492da64ba |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-5b254b59 | sha256-718dc1d356057ca0486712bd3cbaf6ff0474cf10c48d284934fe3c97d8427d8d |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-5b254b59 | sha256-5d60982e8274fe72179ce26aa2294c01ec4428badc627db3a3951e5492da64ba |
