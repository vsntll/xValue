---
name: xValue
description: A big-5 football model on the record, staged as a floodlit night match on black.
colors:
  night: "#08090a"
  night-2: "#0c0d10"
  panel: "#121317"
  panel-2: "#17181d"
  panel-3: "#1d1e24"
  hairline: "rgba(255,255,255,.08)"
  hairline-2: "rgba(255,255,255,.14)"
  ink: "#f5f4fa"
  ink-2: "#b6b8c4"
  ink-3: "#868996"
  cool: "#8b9cff"
  cool-deep: "#4b63ff"
  cool-pale: "#c9d1ff"
  ember: "#b8462a"
  ember-press: "#c24e30"
  ember-hi: "#ff8f66"
  pitch: "#1d5a3a"
typography:
  display:
    fontFamily: "UnifrakturMaguntia, 'Old English Text MT', serif"
    fontSize: "clamp(84px, 11.5vw, 150px)"
    fontWeight: 400
    lineHeight: 0.86
    letterSpacing: "0em"
  display-link:
    fontFamily: "'EB Garamond', Garamond, Georgia, serif"
    fontSize: "clamp(30px, 3.6vw, 46px)"
    fontWeight: 400
    lineHeight: 1
  display-caps:
    fontFamily: "'EB Garamond', Garamond, Georgia, serif"
    fontSize: "clamp(22px, 2.6vw, 34px)"
    fontWeight: 500
    lineHeight: 1
    letterSpacing: "0.12em"
  headline:
    fontFamily: "Geist, system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "clamp(36px, 4.6vw, 60px)"
    fontWeight: 500
    lineHeight: 1.02
    letterSpacing: "-0.035em"
  title:
    fontFamily: "Geist, system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "17px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Geist, system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Geist, system-ui, -apple-system, 'Segoe UI', sans-serif"
    fontSize: "13px"
    fontWeight: 500
    lineHeight: 1.4
  mono:
    fontFamily: "'Geist Mono', ui-monospace, Consolas, monospace"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.4
rounded:
  xs: "4px"
  button: "6px"
  sm: "8px"
  inset: "10px"
  nav: "12px"
  md: "14px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "22px"
  band: "clamp(80px, 11vw, 150px)"
components:
  button-ember:
    backgroundColor: "{colors.ember}"
    textColor: "#ffffff"
    rounded: "{rounded.button}"
    padding: "13px 20px"
    typography: "{typography.label}"
  button-ember-hover:
    backgroundColor: "{colors.ember-press}"
  button-ghost:
    backgroundColor: "rgba(255,255,255,.04)"
    textColor: "{colors.ink}"
    rounded: "{rounded.button}"
    padding: "13px 20px"
  nav-tab:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.sm}"
    padding: "9px 12px"
  nav-tab-active:
    backgroundColor: "{colors.panel-3}"
    textColor: "{colors.ink}"
  chip:
    textColor: "{colors.ink-3}"
    rounded: "{rounded.button}"
    padding: "6px 11px"
  chip-active:
    backgroundColor: "{colors.panel-3}"
    textColor: "{colors.ink}"
  window:
    backgroundColor: "{colors.panel}"
    rounded: "{rounded.md}"
  input-search:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "11px 14px 11px 38px"
---

# Design System: xValue

## Overview

**Creative North Star: "The Night Match"**

xValue is a floodlit match night seen from the players' tunnel. The page is black (#08090a) with a faint 64px chalk grid and a few pinprick stars fixed behind everything. One lit object owns the first viewport: a tall tunnel arch whose rim burns blue on the left and ember on the right, a perspective pitch glowing at its foot, and terrace lettering stacked inside it. Below the arch the product is shown as itself, in dark app windows lit along their top edge by the same blue-white-ember line.

The system runs on light rather than colour. Surfaces are near-black panels separated by hairlines, not fills; the only hues are a cool floodlight blue and a warm ember, and they carry meaning: cool is the model being right or a player being underpriced, ember is the miss or the overprice, and the solid ember fill is reserved for the one action the page wants pressed. Type is three voices: a blackletter for the few display words, an italic and small-caps Garamond for the linking words between them, and Geist for every working line of the product.

Motion is light coming up and focus resolving, never things flying in. Everything enters on a slow exponential ease-out; nothing bounces; reduced motion gets the still page.

**Key Characteristics:**
- Black ground, near-black panels, white hairlines at 8% and 14%.
- Two hues only, cool and ember, each with a fixed meaning.
- Blackletter for display words only; Garamond links them; Geist does the work.
- The blue-white-ember line: arch rim, floor light, window top edge, boot line.
- Dashed edges for anything still live, solid panels for anything frozen.
- Blur-to-sharp focus and rising light on one exponential ease-out.

## Colors

A black night palette with two floodlight hues whose meaning never changes.

### Primary
- **Floodlight Blue** (cool): the model's colour. Correct calls, bargains (model above price), home side in odds bars, the model line in charts, the bold figure in the proof sentence, focus rings, link hover, value bars.
- **Deep Floodlight** (cool-deep): light only, never text. The arch's left side-light, window and panel glows, the team header and match-sim washes, always at low alpha.
- **Pale Floodlight** (cool-pale): the second model series in charts and the assist bars, where a quieter blue is needed beside the main one.

### Secondary
- **Ember** (ember): the single action fill. The hero "See the track record" button and the Track Record nav tab, white text on top. Its pressed/hover/active state is **Ember Press** (ember-press).
- **Ember Glow** (ember-hi): ember as data and light. Misses, overpriced players, away side in odds bars, the price marker on value bars, the arch's right rim and side-light, the input caret.

### Neutral
- **Night** (night): the page ground and every scrollbar track.
- **Night Inset** (night-2): wells inside panels: segmented-control tracks, chip sets, worked examples, detail rows, stat sub-panels.
- **Panel / Panel Raised / Panel Top** (panel, panel-2, panel-3): window and card bodies, inner blocks, and the selected state of tabs and chips.
- **Hairline / Hairline Strong** (hairline, hairline-2): row dividers inside panels, and the 1px ring that draws every panel's edge.
- **Ink / Ink Soft / Ink Quiet** (ink, ink-2, ink-3): headings and figures; body and supporting copy; labels, captions, dates and the neutral "fair" verdict.
- **Pitch** (pitch): the striped grass plane inside the arch only (alternating with #237046).

### Named Rules
**The One Ember Action Rule.** A solid ember fill marks the one action the page wants: see the track record. The hero button and the nav's Track Record tab are the same action, so they share it. Everything else is ghost, panel or text.

**The Two Signals Rule.** Data is cool, ember or neutral, nothing else. Cool means the model was right or the player is worth more than his price; ember-hi means the miss or the overprice; ink-3 means neither. The hit/miss, win/loss and home/away aliases all resolve to these.

**The Light Not Paint Rule.** Cool-deep and ember appear on surfaces only as low-alpha radial light (12 to 42%), never as flat fills behind content.

## Typography

**Display Font:** UnifrakturMaguntia (with 'Old English Text MT', serif)
**Linking Font:** EB Garamond, italic 400 and caps 500 (with Garamond, Georgia, serif)
**Body Font:** Geist 300 to 700 (with system-ui, -apple-system, 'Segoe UI', sans-serif)
**Mono Font:** Geist Mono 400/500 (with ui-monospace, Consolas, monospace)

**Character:** Terrace banner lettering held together by a bookish serif, set against a clean product sans. The blackletter is loud and rare; Geist is quiet and everywhere.

### Hierarchy
- **Display** (UnifrakturMaguntia 400, clamp(84px, 11.5vw, 150px), 0.86): the hero words "Player" and "Price", "Value" in the wordmark (24px hero, 22px nav, up to 112px on the boot screen), team names in the team header, the match-sim score, the footer's outlined word.
- **Display link** (EB Garamond italic 400, clamp(30px, 3.6vw, 46px), 1): "Every" in the hero, the wordmark's x, the italic "v" between fixture teams.
- **Display caps** (EB Garamond 500, clamp(22px, 2.6vw, 34px), uppercase, 0.12em): "HAS A" in the hero.
- **Headline** (Geist 500, clamp(36px, 4.6vw, 60px), 1.02, -0.035em): section intros and page heads.
- **Title** (Geist 600, 17px, 1.3): card and panel headings; window titles at 14px.
- **Body** (Geist 400, 15px, 1.55): all running copy; intros cap at 44 to 72ch. Figures use tabular numerals.
- **Label** (Geist 500, 13 to 13.5px, ink-3): table heads, captions, tile keys, chip text. Sentence case, no tracking.
- **Mono** (Geist Mono 400, 10.5 to 12.5px): dates, xG, minutes, ranks, seasons, code and short tags only.

### Named Rules
**The Display Words Rule.** Blackletter sets single display words and the wordmark, never a sentence, a label or a number in a table. The wordmark x is always the italic Garamond x, because a blackletter x reads as r.

**The Mono Is For Codes Rule.** Geist Mono sets dates, xG, match minutes, ranks, seasons and codes. Prose, names and headline figures stay in Geist.

## Layout

The hero is a full-height (100svh) centred column holding the arch at min(780px, 92vw) wide and min(82svh, 780px) tall, the wordmark pinned top left. Below it, story sections ("bands") sit in a 1200px column with clamp(80px, 11vw, 150px) of air above each; a band head is a two-column grid (1.15fr / 1fr) with the headline left and a short paragraph bottom-aligned right. The working app below the nav lives in a 1240px shell with clamp(14px, 2.4vw, 32px) side padding.

Module grids use auto-fit tracks (340px for cards, 520px for track-record and method panels) with a 16px gap. Panels pad 18 to 22px; rows pad 10 to 12px vertically. At 860px band heads and the proof panel stack to one column; at 640px the arch fills the width with its words pushed to the floor, the stars table becomes two-column cards, all grids go single-column and the window's segmented control hides.

## Elevation & Depth

Depth is light and hairline, not shadow. Every panel is drawn by a 1px inset ring of hairline-2 and sits on one soft drop that only reads against black. Lit objects (the arch, the ember button, the boot line) carry coloured glow; nothing else does.

### Shadow Vocabulary
- **Panel depth** (`box-shadow: 0 0 0 1px rgba(255,255,255,.14), 0 1px 0 rgba(255,255,255,.04) inset, 0 30px 60px -30px rgba(0,0,0,.8)`): every window, card, table wrap and fixture card.
- **Arch glow** (`box-shadow: 0 0 0 1px rgba(255,255,255,.06), 0 0 80px -10px rgba(75,99,255,.35), 0 0 120px -20px rgba(255,120,70,.25)`): the arch only.
- **Ember lift** (`box-shadow: 0 0 0 1px rgba(255,160,120,.25) inset, 0 10px 30px -10px rgba(201,80,47,.7)`): the ember button.
- **Floating nav** (`box-shadow: 0 0 0 1px rgba(255,255,255,.14), 0 20px 40px -20px rgba(0,0,0,.9)` over `backdrop-filter: blur(14px) saturate(1.3)` on rgba(18,19,23,.78)).

### Named Rules
**The Lit Edge Rule.** A window is lit along its top edge by a 1px blue-white-ember line inset 10% each side at 70% opacity, with a 120px cool wash fading down from it. The same three-stop line is the arch rim, the floor light and the boot line.

## Shapes

Generous soft rectangles: 14px for windows and cards, 10px for inset blocks, 8px for tabs, inputs and selects, 6px for buttons and chips, 4px for pills and tags. Player photos and crests are circles. The arch is the one silhouette: a full semicircular top (radius min(390px, 46vw)) over 10px bottom corners, its rim a 2px masked gradient border.

**The Dashed Is Live Rule.** A dashed edge means the numbers can still move: live fixture cards carry a 1px dashed outline inset 7px inside their solid edge. Track-record panels are frozen predictions and stay solid. Dashes elsewhere are only the dotted leaders between a tile's key and value.

## Components

### Buttons
- **Shape:** gently rounded (6px), 13px 20px, Geist 600 at 14.5px.
- **Primary (ember):** ember fill, white text, ember lift. Hover goes to ember-press and rises 1px on the ease-out.
- **Ghost:** 4% white fill with a hairline-2 ring; hover to 8%.
- **Text link:** the hero's "Meet the stars", ink-2 with a down-arrow, ink on hover.

### Chips
- **Style:** a segmented control. A night-2 track (9px radius, 3px padding, hairline ring) holds transparent chips in ink-3.
- **State:** hover to ink; selected chip takes panel-3, ink, and a hairline-2 ring. The window's header segment uses the same form.

### Cards / Containers
- **Corner Style:** 14px.
- **Background:** panel, with night-2 wells inside for sub-blocks.
- **Shadow Strategy:** panel depth (see Elevation).
- **Border:** the inset hairline ring; dividers between rows are hairline.
- **Internal Padding:** 18 to 22px.
- **Fixture card:** live, so dashed (see Shapes); odds bar is three cells, cool / neutral / ember, each a 22% tint with a 40% ring.
- **Track-record card:** solid panel, stat tiles in a night-2 well, first figure in cool.

### Inputs / Fields
- **Style:** panel fill, hairline-2 inset ring, 8px radius, ink-3 placeholder, leading search glyph.
- **Focus:** ring turns cool plus a 4px cool halo at 15%. Caret is ember-hi.

### Navigation
- **Style:** a floating pill (12px radius) that docks 12px from the top once the hero has passed, frosted night glass. Wordmark left, tabs as 14px Geist 500 in ink-2.
- **States:** hover ink on 5% white; active panel-3 with a hairline-2 ring. The Track Record tab is the ember action, always filled, pushed to the end; active adds a 2px ember halo. The two game tabs sit behind a hairline divider under a quiet "Play" label.
- **Mobile:** tabs scroll horizontally with edge fades; the pill tightens.

### Tables
- Label-styled heads on panel, sticky, hairline-2 under them; rows divided by hairline; hover 3% white; open row 8% cool. The sorted column gets an inset 2px underline. Numbers right-aligned and tabular.

### The Arch (signature)
The hero object, layered inside a clipped arch: a dark vertical base (#0b0d16 to #120d0d), cool side-light from the left and ember from the right, a pale floor light, faint tunnel-wall ribs every 48px darkened at the edges, a striped pitch plane tilted at perspective(520px) rotateX(64deg) and masked to fade upward with a white halfway line, and a 2px blue-white-ember floor light with a pale glow. The 2px rim gradient sits above it all. Scrolling fades the light layers to nothing over the first 70vh, leaving the words on black.

### The Proof Panel (signature)
One panel split 0.9fr / 1.1fr: a single held-out sentence in 22 to 30px Geist 500 with its key figure in cool, a quiet citation under it, and beside it a scatter of model value against price on a faint cool wash.

### Motion
- **Ease:** cubic-bezier(.16, 1, .3, 1), exponential ease-out, for every entrance and state change.
- **focusIn:** from opacity 0, blur(14px), scale(1.04). Hero lines resolve one after another at 0.25s, 0.4s, 0.55s, 0.7s, then the sentence (0.9s) and CTA (1.05s); boot word and sim score use it too.
- **rimUp:** the arch rises from brightness(.4) with no glow to full light over 1.4s.
- **settle:** from opacity 0, translateY(8px); sections, detail rows, reveals.
- **grow:** scaleX from 0 on bars and the boot line.
- **Reduced motion:** every animation and transition off; the page is simply lit.

## Do's and Don'ts

### Do:
- **Do** keep the solid ember fill for the track-record action alone (hero button and Track Record tab).
- **Do** colour data with cool for right or underpriced, ember-hi for miss or overpriced, ink-3 for neither.
- **Do** set display words in UnifrakturMaguntia, link them with italic or caps EB Garamond, and set everything else in Geist.
- **Do** draw panels with the 1px hairline-2 ring and panel depth, and light window tops with the blue-white-ember line.
- **Do** give live, still-moving cards a dashed inset outline and keep frozen records solid.
- **Do** bring things in with focusIn, rimUp, settle or grow on cubic-bezier(.16, 1, .3, 1), and stop all of it under reduced motion.
- **Do** keep Geist Mono to dates, xG, minutes, ranks, seasons and codes.

### Don't:
- **Don't** fill text with a gradient; the three-stop line is light on edges, never in letters.
- **Don't** introduce a third data hue, or use cool or ember for decoration where they would read as a verdict.
- **Don't** set sentences, labels or table figures in blackletter, or use a blackletter x in the wordmark.
- **Don't** use flat cool or ember fills behind content; they appear as low-alpha light.
- **Don't** add bounce, overshoot or travel to motion; light rises and focus resolves.
