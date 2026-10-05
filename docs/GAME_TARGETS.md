# Exact game targets and measurable success

The potato target was corrected in the live client on 5 October 2026 after the user identified the intended game. Game descriptions establish objectives; they are not evidence that this model can play. The action vocabularies in `configs/` are provisional until their controls and sensitivity are confirmed in the live client.

| Target | Place ID | Objective used in this repository |
| --- | --- | --- |
| [Peel THE Potato — soviet potato republic](https://www.roblox.com/games/116701845804918/Peel-THE-Potato) | `116701845804918` | Find the hidden key, capture its visible evidence, unlock the vault, and show the escape outcome. |
| [Deadly Delivery — WTHHHELL BRO](https://www.roblox.com/games/125810438250765/Deadly-Delivery) | `125810438250765` | First benchmark: evacuate from floor 10 after a fresh run. This is a milestone, not a claim of total game completion. |

The live-tested potato loop is collecting potatoes, depositing them into the raw crate, peeling with mouse swipes at a table, selling at the peeled-potato crate, and purchasing tools with earned game currency. The visible 2,500-potato quest is a milestone; it is not proof of finding the key or escaping. External keyboard/mouse development is in progress and no trained policy has completed this target.

The [official Deadly Delivery page with badge descriptions](https://www.roblox.com/es/games/125810438250765/Deadly-Delivery?gameSearchSessionInfo=9466c5e1-25b8-4693-8cfc-9cddb6fcfb17&isAd=false&nativeAdData=&numberOfLoadedTiles=103&page=searchPage&placeId=125810438250765&position=85&universeId=8950496606) describes evacuation badges for floors 10, 20 and 30. Freeze a specific start and evacuation goal before testing. Higher-floor achievements, cooperative survival and a complete solo run are separate tasks with separate denominators.

The earlier `peel_a_potato.json` profile refers to a different game by Apartment Horrors. Use `peel_the_potato.json` for the current target. The generic obby config does not certify a particular obby.

## What counts as a trial

Start each trial with the same checkpoint, declared model SHA-256, action config, window size and camera sensitivity. Record the game place, account progress, equipped upgrades, chosen route, party size, start condition and criterion before play. Progress saved from previous shipments can make later trials easier; identify it rather than treating all runs as equivalent.

A trial begins when the agent receives control in the declared start state. All subsequent deaths, timeouts, disconnects, stuck loops, capture failures and manual interventions belong in the outcome. A retry is another trial. Manual setup is permitted before the trial; any intervention after it starts makes that trial assisted. Do not discard difficult starts or only report the best clip.

For the current potato target, preserve the initial progress and upgrades, the key-found screenshot, and the actual vault escape outcome. The development session began cooperatively before continuing solo; it cannot establish a fresh solo full-game win.

For a delivery evacuation trial, preserve the fresh-run start, progression and actual evacuation result. A lobby screen, loading screen, survived minute, delivery interaction, or floor counter by itself is not proof of evacuation. If another player carried the objective, report the cooperative contribution explicitly; it is not solo-agent completion.

## Long sessions and reliability

Use early runs for development. After freezing a candidate checkpoint, run a separate evaluation set that was not used to select demonstrations or tune controls. Report the exact number of trials and all outcomes, plus a 95% Wilson interval. For example, even 1/1 successes gives a lower bound of only about 20.7%; it does not establish reliable play.

A reasonable declared target is 30 fresh independent trials per criterion and at least one continuous 60-minute run with no manual recovery. These are acceptance targets, not measured results. Report valid completed cycles per minute, deaths, stuck events, interventions, longest unassisted episode, end-to-end decision latency, and peak policy-process memory. Keep incomplete runs in the denominator. Hour-long action output without visible task progress does not meet sustained-play success.

The evaluator checks complete logs, screenshot decoding, file hashes, checkpoint consistency and reviewer annotations. It cannot determine from pixels whether the reviewer was correct. `verified_wins` means *reviewed visible successes with the required files*, not a cryptographic or automated proof of gameplay. Keep the full episode evidence for independent review.
