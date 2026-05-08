/**
 * UCLA Anderson – PptxGenJS Style Module
 * ========================================
 * Reusable style constants, design philosophy, and helper functions
 * for building MGMT298D lecture slides with PptxGenJS.
 *
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                  CONTENT DESIGN PHILOSOPHY                      │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * These are the audience, purpose, and the principles that should
 * guide all content decisions for this slide deck. They are written
 * abstractly so they can be applied to any week's topic.
 *
 *
 * 1. AUDIENCE & PURPOSE
 *    ───────────────────
 *    The audience is MBA students. They are analytically capable but
 *    not necessarily technical specialists. Slides serve as the
 *    primary learning artifact — students will revisit them after
 *    class. Every slide should therefore be self-contained enough
 *    to be understood without the instructor's narration, but concise
 *    enough not to be a wall of text.
 *
 *
 * 2. ONE IDEA PER SLIDE
 *    ───────────────────
 *    Each slide should convey a single concept, comparison, or
 *    takeaway. If you find yourself needing two headings or two
 *    unrelated bullet groups, split into two slides. White space
 *    is not wasted space — it signals clarity.
 *
 *
 * 3. TABLES, CARDS, AND OTHER LAYOUTS
 *    ──────────────────────────────────
 *    Tables are welcome on this deck. Weeks 1–2 use them effectively
 *    for showing how numbers move across conditions — e.g., how LASSO
 *    coefficients change as λ grows, Ridge vs LASSO side-by-side, or
 *    raw vs scaled feature values. When the point of the slide is
 *    "watch how these numbers shift", a clean table beats prose.
 *
 *    Table conventions actually used:
 *      - Blue header row, white text, alternating light-grey body rows.
 *      - Red highlighting on cells that illustrate the slide's point
 *        (e.g., coefficients zeroed out by LASSO).
 *      - Captioned by one short line of prose above or below.
 *
 *    For data shapes that are NOT a "watch the numbers shift" story,
 *    choose a layout that matches the shape of the data:
 *
 *    ┌──────────────────────────┬─────────────────────────────────────────────┐
 *    │ DATA SHAPE               │ PREFERRED LAYOUT                           │
 *    ├──────────────────────────┼─────────────────────────────────────────────┤
 *    │ Numbers shifting across  │ Plain table with blue header + red         │
 *    │ conditions               │ highlights on the cells you care about     │
 *    ├──────────────────────────┼─────────────────────────────────────────────┤
 *    │ Parallel categories with │ Color-topped cards (addTopCard)            │
 *    │ the same attributes      │ — see Regularization Summary slide         │
 *    ├──────────────────────────┼─────────────────────────────────────────────┤
 *    │ Quantities, proportions, │ Horizontal bars (addHBar)                  │
 *    │ probabilities            │ — see Feature Importance slide             │
 *    ├──────────────────────────┼─────────────────────────────────────────────┤
 *    │ Sequential process or    │ Numbered flow (addBullets with numbering)  │
 *    │ pipeline steps           │ — see "A Standard ML Pipeline" slide       │
 *    ├──────────────────────────┼─────────────────────────────────────────────┤
 *    │ Side-by-side contrast    │ Two-column layout with bold subheadings    │
 *    │ (A vs B)                 │ — see "Trees vs Linear Regression" slide   │
 *    ├──────────────────────────┼─────────────────────────────────────────────┤
 *    │ Term → definition        │ Inline: bold-blue term + colon + plain     │
 *    │                          │ gloss in running text (NOT a row layout)   │
 *    ├──────────────────────────┼─────────────────────────────────────────────┤
 *    │ A handful of short ideas │ Blank-line-separated lines of body text    │
 *    │                          │ (no bullet markers) — the dominant style   │
 *    └──────────────────────────┴─────────────────────────────────────────────┘
 *
 *    When in doubt, ask: "What do I want the student's eye to land
 *    on first?" Then make that element the largest or most colorful
 *    thing on the slide.
 *
 *
 * 4. VISUAL HIERARCHY
 *    ─────────────────
 *    Every content slide has three tiers of importance, expressed
 *    through size, weight, and color:
 *
 *    Tier 1 – Heading:    Large, bold, primary blue. Whitney Bold 28pt.
 *             The student should be able to skim headings alone and
 *             reconstruct the lecture's narrative arc.
 *
 *    Tier 2 – Key term:   Inline bold-blue treatment of the term being
 *             defined or the concept being introduced — sitting inside
 *             a normal line of body text. This is the dominant Tier-2
 *             pattern in Weeks 1–2 (e.g., "**Tree-based models**:
 *             Alternative to linear regression for predictions").
 *             Stat cards exist as a helper but are used sparingly.
 *
 *    Tier 3 – Supporting: Regular-weight black body text. Descriptions,
 *             caveats, examples, the rest of the line after the bolded
 *             term. The slide should still make sense if you squint
 *             and only read Tiers 1 and 2.
 *
 *
 * 5. CLOSING THE SLIDE — INLINE PUNCH LINES, NOT FOOTERS
 *    ────────────────────────────────────────────────────
 *    Weeks 1–2 do not use a dedicated footer "takeaway" zone. Instead,
 *    the "so what" of a slide lives inline as the last line of body
 *    text — usually as a punchy fragment, a rhetorical question, or
 *    a short call to action. Examples:
 *      - "First ML model? Linear regression!"
 *      - "How do we find the best weights?"
 *      - "Can we do better?"
 *      - "Most predictive features survive"
 *      - "Does not fully eliminate variables from model"
 *
 *    The addFooter() helper still exists for cases where a visually
 *    distinct closing line is wanted, but it should be the exception
 *    rather than the rule.
 *
 *
 * 6. LEVEL OF DETAIL
 *    ────────────────
 *    Strike a balance between two failure modes:
 *
 *    Too sparse: Slides that just say "Temperature" with no context.
 *    The student reviews later and can't reconstruct the point.
 *
 *    Too dense: Slides that read like a textbook paragraph. The
 *    student zones out mid-slide and retains nothing.
 *
 *    The sweet spot for a content slide (matches Weeks 1–2):
 *      - A clear heading (3–6 words)
 *      - 3–4 short body lines, each 1–2 lines wrapped, separated
 *        by blank space — NOT bullet markers
 *      - Optional inline punch line as the last line (see §5)
 *      - Optional small decorative image in the bottom-right corner
 *
 *    If a slide ever needs more than ~5 lines of body text, split it
 *    into two slides instead.
 *
 *
 * 7. CONSISTENCY ACROSS THE DECK
 *    ────────────────────────────
 *    The same type of data should always get the same visual
 *    treatment. If slide 8 shows model parameters as stat cards,
 *    slide 25 should not show a different set of parameters as a
 *    table. This consistency trains the student's eye: "Ah, cards
 *    again — these are parameter settings I should know."
 *
 *    Use the helper functions in this module to enforce consistency
 *    mechanically. If you need a new layout type, add it here as a
 *    shared helper rather than inlining one-off code in the slide
 *    script.
 *
 *
 * 8. DECK STRUCTURE
 *    ───────────────
 *    Each week's deck follows this skeleton (matches Weeks 1–2):
 *
 *      1. Title slide        — course, week, topic, author
 *      2. "Today's Class"    — 1 short framing slide, 2–3 lines of
 *                              prose, no bullets. Optional and very
 *                              brief — Week 2 uses one, Week 1 jumps
 *                              straight into content.
 *      3. Section divider    — full-blue background, large bold white
 *                              Whitney title, nothing else
 *      4. Content slides     — the actual teaching material, 3–4
 *                              lines per slide, see §6
 *      5. Break / interlude  — playful full-blue slide with diagonal
 *                              light-blue title and a small mascot
 *                              illustration; used roughly once per
 *                              lecture for the mid-class break
 *      6. Repeat 3–4 for each section
 *
 *    There is no closing "summary / key takeaways" slide in the
 *    Week 1–2 decks. Each section's punch line lives inline on its
 *    own slide instead (see §5). Aim for 3–5 section dividers per
 *    week's lecture.
 *
 *
 * 9. (Reserved — language and voice rules now live in §10.)
 *
 *
 * 10. LANGUAGE & VOICE (derived from Weeks 1–2 decks)
 *    ─────────────────────────────────────────────────
 *    The instructor's voice across the deck is conversational,
 *    intuition-first, and lightly playful. Match these patterns when
 *    drafting any new slide text.
 *
 *    A. Sentence shape
 *       - Short, standalone lines separated by blank space (NOT bullets
 *         with a • marker). Each line is one idea. 1–2 lines per idea.
 *       - TIGHT. Target ~12 words per line, hard cap ~18. If a line
 *         runs longer, cut 3–5 words — drop articles, "that", "which",
 *         "in order to", redundant qualifiers, hedges.
 *       - Drop articles and verbs where natural: "Branch of artificial
 *         intelligence focused on prediction" / "Mostly uses structured,
 *         tabular data" / "No phones".
 *       - Fragments are preferred over full sentences when they carry
 *         the same meaning. Telegraphic phrasing is good
 *         ("Three problems", "Result: 30+ FPS", "A quarter the pixels").
 *       - Full sentences are fine when they add warmth ("We will often
 *         look at Python code together…").
 *       - 3–4 lines per content slide is typical. Almost never more
 *         than 5.
 *
 *    B. Definition pattern (very common)
 *       Bold term + colon + plain-English gloss, often followed by an
 *       example on the next line.
 *           **Feature engineering**: Use existing features to create
 *           new features!
 *           **Overfitting:** Model is too complex for available data,
 *           fits to random noise instead of extracting real signal
 *
 *    C. Framing devices the instructor reuses
 *       - Use SPARINGLY. Most slides need NO framing label at all —
 *         just declarative lines.
 *       - Occasional, when truly useful:
 *           "Example: …"        — to ground an abstraction
 *           "Whenever you hear X think Y" — for vocabulary anchoring
 *       - Rhetorical questions as transitions / hooks:
 *           "How do we find the best weights?"
 *           "First ML model? Linear regression!"
 *           "Can we do better?"
 *           "What other new features would make sense for H&M?"
 *       - DO NOT USE these LinkedIn-style framing labels:
 *           "Main idea:", "Key idea:", "Intuition:", "Problem:",
 *           "The fix:", "The shift:", "The old way:", "The new way:",
 *           "TL;DR:", "Bottom line:". They feel buzzy and corporate.
 *
 *    D. Tone
 *       - Warm, first-person plural: "we'll cover", "we will skip
 *         details for now", "we care much less about p-values".
 *       - Lightly enthusiastic, with occasional exclamation marks on
 *         payoff lines ("just want to make good predictions!",
 *         "Linear regression!").
 *       - Mild self-deprecation / honesty about scope: "Lots of Python
 *         packages etc. to handle this — we will skip details for now".
 *       - Casual asides in parentheses or italics for color:
 *         "(Phones and laptops away please!)", "*alternative to linear
 *         regression*".
 *
 *    E. Punctuation & typography habits
 *       - Em dash (—) and en dash (–) used liberally for asides and
 *         appositive definitions. Long arrow (→) for cause/effect.
 *       - Equals sign as shorthand: "Each leaf = a prediction",
 *         "Training linear regression = finding model weights".
 *       - "Scare quotes" around informal or borrowed terms: "gold
 *         standard", "weight-free", "wisdom of the crowds", "if
 *         then", model "weights".
 *       - Italics for first introduction of a technical term
 *         (*overfitting*, *training examples*, *neural network*).
 *       - Bold for the term being defined on that slide.
 *       - Parentheses for terse clarifications: "(predicting a number)",
 *         "(memorization vs learning)", "(a positive number)".
 *
 *    F. Vocabulary anchoring
 *       Whenever a new term is introduced, immediately follow it with
 *       a plain-language equivalent or a "what to think when you hear
 *       this" line. The goal stated on the deck itself is to
 *       "massively expand your vocabulary around ML and AI", so each
 *       slide should leave the student with at least one word they
 *       can now use confidently.
 *
 *    G. What to AVOID in language
 *       - No marketing-speak ("revolutionary", "cutting-edge",
 *         "leverage synergies").
 *       - No long compound sentences with multiple clauses.
 *       - No textbook-style passive voice ("It can be observed
 *         that…"). Prefer active and direct.
 *       - No bullet markers (•) on simple content slides — use blank
 *         lines between thoughts instead. Reserve bullets for true
 *         enumerated lists.
 *       - Avoid more than one definition per slide; if there's a
 *         second term, it gets its own slide.
 *
 *
 * 11. SLIDE-TYPE STYLE PRESETS (from Week 2 reference)
 *    ───────────────────────────────────────────────────
 *    Three reusable slide "moods" appear throughout the decks:
 *
 *    a. CONTENT slide  (white bg)
 *       - Left-aligned blue Whitney Bold title + thin grey rule.
 *       - Thin blue accent bar on the far left edge of the slide.
 *       - 3–4 short black sans-serif lines separated by blank space.
 *       - Optional small decorative photo / illustration tucked in
 *         the bottom-right corner.
 *       - Inline bold-blue treatment for the term being defined.
 *
 *    b. SECTION DIVIDER  (full blue bg)
 *       - Dark blue fill across the whole slide.
 *       - Centered, large bold WHITE Whitney Bold title.
 *       - Nothing else on the slide. Used to mark a new section.
 *       - Variant: gold title instead of white for an extra-major
 *         section break (used sparingly).
 *
 *    c. BREAK / INTERLUDE  (full blue bg, playful)
 *       - Dark blue fill.
 *       - Large light-blue title set on a slight diagonal angle.
 *       - Small 3D illustration (sleeping robot, etc.) in the
 *         bottom-right corner.
 *       - Used for "15-min Break" and similar in-class interludes.
 *
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                        API USAGE                                │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * Usage:
 *   const pptxgen = require("pptxgenjs");
 *   const style = require("./ucla-anderson-style");
 *
 *   const pres = new pptxgen();
 *   style.initPresentation(pres, {
 *     author: "Auyon Siddiq",
 *     title:  "MGMT298D Week 8 – Topic Name",
 *   });
 *
 *   // Title slide
 *   style.addTitleSlide(pres, {
 *     courseCode:  "MGMT298D",
 *     courseName:  "Science and Strategy of AI",
 *     weekLabel:   "Week 8",
 *     topicTitle:  "Topic Name",
 *     institution: "UCLA Anderson School of Management",
 *     author:      "Auyon Siddiq",
 *   });
 *
 *   // Section divider
 *   style.addDivider(pres, "Section Title");
 *
 *   // Content slide with heading + bullets + footer
 *   const slide = pres.addSlide();
 *   style.addTitle(pres, slide, "Slide Heading");
 *   style.addBullets(slide, [
 *     "First point",
 *     "Second point",
 *   ]);
 *   style.addFooter(pres, slide, "Key takeaway text here.");
 *
 *   // Stat cards (for parameters / numeric values)
 *   style.addStatCard(pres, slide, 0.5, 1.4, 2.8, 1.0,
 *     "Label", "Big Value", "Optional description");
 *
 *   // Accent rows (for term → definition lists)
 *   style.addAccentRow(pres, slide, 1.3, "Label", "Description text");
 *
 *   // Color-topped cards (for parallel category comparison)
 *   style.addTopCard(pres, slide, 0.5, 1.2, 2.5, 3.0, "005587",
 *     "Category A", [{label:"Metric", value:"42"}], "Footnote");
 *
 *   // Horizontal bars (for probabilities / proportions)
 *   style.addHBar(pres, slide, 2.0, 2.5, 5.0, 0.25, 0.73);
 *
 *   pres.writeFile({ fileName: "output.pptx" });
 */

// ═══════════════════════════════════════════════════
// COLOR PALETTE  (extracted from UCLA Anderson template XML)
// ═══════════════════════════════════════════════════
const C = {
  blue:     "005587",   // Primary UCLA blue – titles, headers, accent bars
  blue2:    "2774AE",   // Secondary / lighter blue
  gold:     "FFD100",   // Gold – title-slide accents
  darkGold: "C4820E",   // Dark gold – optional emphasis
  body:     "000000",   // Body text (pure black — no dark gray)
  muted:    "000000",   // Secondary text (black — no light gray)
  altRow:   "EDF2F7",   // Alternating-row background
  footerBg: "EBF5FB",   // Footer callout-bar background (unused — footers removed)
  lineSep:  "E8E8E8",   // Thin separator line under titles
  border:   "CCCCCC",   // Card / table borders
  white:    "FFFFFF",
  red:      "CC0000",
  green:    "2E8B57",
  orange:   "D4740A",
};

// ═══════════════════════════════════════════════════
// TYPOGRAPHY
// ═══════════════════════════════════════════════════
const TITLE_FONT = "Whitney Bold";   // slide headings & section dividers
const BODY_FONT  = "Arial";          // body text, bullets, labels, footers
const TITLE_SIZE = 28;               // pt – main slide heading
const BODY_SIZE  = 14;               // pt – body / bullet text
const SMALL_SIZE = 12;               // pt – footer callout, sub-labels

// ═══════════════════════════════════════════════════
// LAYOUT CONSTANTS
// ═══════════════════════════════════════════════════
const LEFT_BAR_W    = 0.06;          // thin accent bar width (inches)
const CONTENT_X     = 0.5;           // left margin for content
const CONTENT_W     = 9.1;           // content width
const TITLE_Y       = 0.2;           // title top position
const SEPARATOR_Y   = 0.98;          // line separator Y
const FOOTER_Y      = 4.85;          // default footer Y
const FOOTER_H      = 0.5;           // footer bar height
const CARD_BORDER_W = 0.75;          // card border line width (pt)

// ═══════════════════════════════════════════════════
// PRESENTATION INIT
// ═══════════════════════════════════════════════════
function initPresentation(pres, opts = {}) {
  pres.layout = "LAYOUT_16x9";
  if (opts.author) pres.author = opts.author;
  if (opts.title)  pres.title  = opts.title;
}

// ═══════════════════════════════════════════════════
// SLIDE HELPERS
// ═══════════════════════════════════════════════════

/** Thin blue vertical bar on the left edge of content slides. */
function addLeftBar(pres, slide) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: LEFT_BAR_W, h: 5.63,
    fill: { color: C.blue },
  });
}

/**
 * Slide heading (Whitney Bold, blue) + separator line.
 * Automatically adds the left accent bar.
 */
function addTitle(pres, slide, title) {
  addLeftBar(pres, slide);
  slide.addText(title, {
    x: CONTENT_X, y: TITLE_Y, w: CONTENT_W, h: 0.75,
    fontSize: TITLE_SIZE, fontFace: TITLE_FONT, bold: true,
    color: C.blue, margin: 0,
  });
  slide.addShape(pres.shapes.LINE, {
    x: CONTENT_X, y: SEPARATOR_Y, w: CONTENT_W, h: 0,
    line: { color: C.lineSep, width: 1 },
  });
}

/**
 * Footer callout — plain bold blue text, no background bar.
 * States the "so what" of the slide in a single sentence.
 * @param {number} [y=4.85] – optional Y override
 */
function addFooter(pres, slide, text, y) {
  const barY = y || FOOTER_Y;
  slide.addText(text, {
    x: 0.55, y: barY, w: 8.9, h: FOOTER_H,
    fontSize: SMALL_SIZE, fontFace: BODY_FONT, bold: true,
    color: C.blue, margin: 0, valign: "middle",
  });
}

/**
 * Stat callout card – white box with thin border + blue left accent bar.
 * Great for replacing table cells with visually distinct cards.
 *
 * @param {number} x,y,w,h – position & size in inches
 * @param {string} label   – small blue header text
 * @param {string} value   – large bold value
 * @param {string} [desc]  – optional muted description below value
 */
function addStatCard(pres, slide, x, y, w, h, label, value, desc) {
  // Card background + border
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.white },
    line: { color: C.border, width: CARD_BORDER_W },
  });
  // Blue left accent
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: LEFT_BAR_W, h,
    fill: { color: C.blue },
  });
  // Label
  slide.addText(label, {
    x: x + 0.18, y: y + 0.06, w: w - 0.3, h: 0.26,
    fontSize: 10, fontFace: BODY_FONT, bold: true,
    color: C.blue, margin: 0,
  });
  // Value
  slide.addText(value, {
    x: x + 0.18, y: y + 0.3, w: w - 0.3, h: 0.28,
    fontSize: 15, fontFace: BODY_FONT, bold: true,
    color: C.body, margin: 0,
  });
  // Description
  if (desc) {
    slide.addText(desc, {
      x: x + 0.18, y: y + 0.56, w: w - 0.3, h: 0.4,
      fontSize: 9.5, fontFace: BODY_FONT, color: C.muted, margin: 0,
    });
  }
}

/**
 * Section divider – full blue background, centered white Whitney Bold text.
 * Returns the created slide.
 */
function addDivider(pres, title) {
  const slide = pres.addSlide();
  slide.background = { color: C.blue };
  slide.addText(title, {
    x: 1, y: 1.5, w: 8, h: 2.5,
    fontSize: 36, fontFace: TITLE_FONT, bold: true,
    color: C.white, align: "center", valign: "middle",
  });
  return slide;
}

/**
 * Title slide – course branding, week label, topic, institution, author.
 * Returns the created slide.
 */
function addTitleSlide(pres, opts = {}) {
  const {
    courseCode  = "MGMT298D",
    courseName  = "Science and Strategy of AI",
    weekLabel   = "",
    topicTitle  = "",
    institution = "UCLA Anderson School of Management",
    author      = "Auyon Siddiq",
  } = opts;

  const s = pres.addSlide();
  s.background = { color: C.blue };

  // Course code (gold)
  s.addText(courseCode, {
    x: 0, y: 0.7, w: 10, h: 0.45,
    fontSize: 18, fontFace: BODY_FONT, color: C.gold,
    align: "center", margin: 0,
  });
  // Course name (gold)
  s.addText(courseName, {
    x: 0, y: 1.1, w: 10, h: 0.4,
    fontSize: 16, fontFace: BODY_FONT, color: C.gold,
    align: "center", margin: 0,
  });
  // Week label (white, bold)
  if (weekLabel) {
    s.addText(weekLabel, {
      x: 0, y: 1.8, w: 10, h: 0.7,
      fontSize: 22, fontFace: TITLE_FONT, bold: true, color: C.white,
      align: "center", margin: 0,
    });
  }
  // Topic title (white, large bold)
  if (topicTitle) {
    s.addText(topicTitle, {
      x: 0, y: 2.5, w: 10, h: 0.9,
      fontSize: 36, fontFace: TITLE_FONT, bold: true, color: C.white,
      align: "center", margin: 0,
    });
  }
  // Institution (white, small)
  s.addText(institution, {
    x: 0, y: 3.8, w: 10, h: 0.35,
    fontSize: 12, fontFace: BODY_FONT, color: C.white,
    align: "center", margin: 0,
  });
  // Author (white, bold)
  s.addText(author, {
    x: 0, y: 4.15, w: 10, h: 0.35,
    fontSize: 14, fontFace: BODY_FONT, bold: true, color: C.white,
    align: "center", margin: 0,
  });
  return s;
}

/**
 * Bullet-point list in body area.
 * @param {Array<string|{text,bold,color}>} items – plain strings or rich objects
 * @param {object} [opts] – override x, y, w, h, fontSize
 */
function addBullets(slide, items, opts = {}) {
  const rows = items.map((item) => {
    const isObj = typeof item === "object";
    return {
      text: isObj ? item.text : item,
      options: {
        fontSize:  opts.fontSize || BODY_SIZE,
        fontFace:  BODY_FONT,
        color:     (isObj && item.color) || C.body,
        bold:      isObj ? !!item.bold : false,
        bullet:    { type: "bullet" },
        paraSpaceAfter: 6,
      },
    };
  });
  slide.addText(rows, {
    x: opts.x || CONTENT_X, y: opts.y || 1.15,
    w: opts.w || CONTENT_W, h: opts.h || 3.5,
    valign: "top", margin: [0, 0, 0, 6],
  });
}

/**
 * Accent-bar row – blue left bar + bold label + description to the right.
 * Useful for "swimlane" / key-value layouts that replace table rows.
 *
 * @param {number} y        – vertical position (inches)
 * @param {string} label    – bold blue label (left column)
 * @param {string} desc     – body-color description (right column)
 * @param {object} [opts]   – override labelW, rowH, startX
 */
function addAccentRow(pres, slide, y, label, desc, opts = {}) {
  const startX = opts.startX || CONTENT_X;
  const labelW = opts.labelW || 2.8;
  const rowH   = opts.rowH   || 0.55;

  // Blue accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: startX, y, w: LEFT_BAR_W, h: rowH,
    fill: { color: C.blue },
  });
  // Label
  slide.addText(label, {
    x: startX + 0.18, y, w: labelW, h: rowH,
    fontSize: BODY_SIZE, fontFace: BODY_FONT, bold: true,
    color: C.blue, margin: 0, valign: "middle",
  });
  // Description
  slide.addText(desc, {
    x: startX + 0.18 + labelW + 0.15, y,
    w: CONTENT_W - labelW - 0.5, h: rowH,
    fontSize: BODY_SIZE, fontFace: BODY_FONT,
    color: C.body, margin: 0, valign: "middle",
  });
}

/**
 * Color-topped card – rectangular card with a colored band at the top.
 * Used for side-by-side comparison / category cards (replaces table columns).
 *
 * @param {string} topColor – hex color for the top accent band
 * @param {string} heading  – bold card heading
 * @param {Array<{label,value}>} rows – key-value pairs inside the card
 * @param {string} [footnote] – italic muted text at the bottom
 */
function addTopCard(pres, slide, x, y, w, h, topColor, heading, rows, footnote) {
  // Card body
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.white },
    line: { color: C.border, width: CARD_BORDER_W },
  });
  // Colored top band
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 0.08,
    fill: { color: topColor },
  });
  // Heading
  slide.addText(heading, {
    x: x + 0.12, y: y + 0.12, w: w - 0.24, h: 0.35,
    fontSize: 13, fontFace: BODY_FONT, bold: true,
    color: C.blue, align: "center", margin: 0,
  });
  // Separator
  slide.addShape(pres.shapes.LINE, {
    x: x + 0.12, y: y + 0.48, w: w - 0.24, h: 0,
    line: { color: C.lineSep, width: 0.5 },
  });
  // Key-value rows
  let rowY = y + 0.55;
  for (const row of rows) {
    slide.addText(row.label, {
      x: x + 0.12, y: rowY, w: w - 0.24, h: 0.22,
      fontSize: 9, fontFace: BODY_FONT, color: C.muted,
      align: "center", margin: 0,
    });
    slide.addText(row.value, {
      x: x + 0.12, y: rowY + 0.18, w: w - 0.24, h: 0.3,
      fontSize: 16, fontFace: BODY_FONT, bold: true, color: C.body,
      align: "center", margin: 0,
    });
    // Thin separator between rows
    slide.addShape(pres.shapes.LINE, {
      x: x + 0.2, y: rowY + 0.5, w: w - 0.4, h: 0,
      line: { color: C.lineSep, width: 0.5 },
    });
    rowY += 0.55;
  }
  // Footnote
  if (footnote) {
    slide.addText(footnote, {
      x: x + 0.12, y: y + h - 0.45, w: w - 0.24, h: 0.35,
      fontSize: 9, fontFace: BODY_FONT, italic: true,
      color: C.muted, align: "center", margin: 0,
    });
  }
}

/**
 * Horizontal bar (e.g., for probability or percentage visualizations).
 *
 * @param {number} x,y,w,h – bar position and max dimensions
 * @param {number} pct     – fill percentage (0–1)
 * @param {string} [color] – fill color (defaults to primary blue)
 */
function addHBar(pres, slide, x, y, w, h, pct, color) {
  // Background track
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.altRow },
  });
  // Filled portion
  if (pct > 0) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y, w: w * Math.min(pct, 1), h,
      fill: { color: color || C.blue },
    });
  }
}

// ═══════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════
module.exports = {
  // Constants
  C,
  TITLE_FONT,
  BODY_FONT,
  TITLE_SIZE,
  BODY_SIZE,
  SMALL_SIZE,

  // Layout constants
  LEFT_BAR_W,
  CONTENT_X,
  CONTENT_W,
  TITLE_Y,
  SEPARATOR_Y,
  FOOTER_Y,
  FOOTER_H,
  CARD_BORDER_W,

  // Functions
  initPresentation,
  addLeftBar,
  addTitle,
  addFooter,
  addStatCard,
  addDivider,
  addTitleSlide,
  addBullets,
  addAccentRow,
  addTopCard,
  addHBar,
};
