# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fe871fc756965ed2ad10891f894211093b77d588ff529c17002e62b815bc7e20`
- fixture hash: `sha256-a80c2f2bd19aa96d2482ebad0e531e3744f53fc1802236a8f23f6e8e699cab58`
- score hash: `sha256-df2b68c28abe6879f899b42204ba068638f751f7c18bfd47bf0235537d1a17f3`
- bundle hash: `sha256-669936e90220ebb6e0cf06562e4beba31a33f42fcb1afb746384841c39f8303e`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ce2ae1dd4e3d2b3f6fd3a8f55230aa6172c7dcefb57f249ba3ae3221dcb1a66b |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-5a4af17f1ba1bbf4c33292e66139a1a098e573e39eb769214770a19428d353d9 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b77c25968d61ccda19db42686dc76f1879959869f1614017cc95b4e5afec36a3 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-b0039adc7f655cc524fa36b055f92ff4ff359adfb1cda5e23cf0e849cdbbc9b0 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f9b44aea | sha256-ded24423ce073436784777da5b4bf89f73a5d13eb4c56d4118a508c25a15c50d |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-f9b44aea | sha256-1d35c788b3d2b95a04156bebc922277b704072395979aed0ce6da8e9298d1476 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-7dda18c9 | sha256-93627f75f6b0181499b24c7ef954208d1fea86a79630c6d3c5d51caf7665fcc7 |
