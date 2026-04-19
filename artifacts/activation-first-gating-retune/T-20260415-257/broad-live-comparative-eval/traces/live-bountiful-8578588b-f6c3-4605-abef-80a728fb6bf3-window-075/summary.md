# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-075`
- winner mode: `graph_prior_only`
- trace hash: `sha256-fe871fc756965ed2ad10891f894211093b77d588ff529c17002e62b815bc7e20`
- fixture hash: `sha256-a80c2f2bd19aa96d2482ebad0e531e3744f53fc1802236a8f23f6e8e699cab58`
- score hash: `sha256-0a78064305e2eb58e2487292f507e5f8c9ffd0b1a95896aa6da01714ee5f52b0`
- bundle hash: `sha256-6223077caa2738af518a55fd85d6a63fa74001fdf9bdd0537a6343067d7718ac`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ce2ae1dd4e3d2b3f6fd3a8f55230aa6172c7dcefb57f249ba3ae3221dcb1a66b |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-3fee0c9c3a1727facad68efa072ba93f940a20ba50cc1d01dc8002e85abe23c9 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-7aca23b12abf74e296750ba21fe6a06de2f26f9b1359c0c26d8bcdecf45bfb42 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-8c9f9edc2476148bf2714fc42c86e007d4136e49070a248704d004b3b5263d83 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7dda18c9 | sha256-0d5a6fc6d767784c60c4a44791277c5055d7a09d8da6223e749e16ac791c3b02 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-7dda18c9 | sha256-6e01d3ec686bf50843f60a7eb16711bc11cc95398686d3a6d657c508e90e5bf1 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7dda18c9 | sha256-0d5a6fc6d767784c60c4a44791277c5055d7a09d8da6223e749e16ac791c3b02 |
