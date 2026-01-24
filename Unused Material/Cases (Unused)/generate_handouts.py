"""
Generate PDF Case Study Handouts for MGMT298D
UCLA Anderson - Science and Strategy of AI

Creates:
1. Student handouts (no key insights - those emerge from discussion)
2. Instructor notes (with teaching guidance and key insights)
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# UCLA Colors
UCLA_BLUE = HexColor('#2774AE')
UCLA_DARK_BLUE = HexColor('#005587')
UCLA_GOLD = HexColor('#FFD100')
DARK_GRAY = HexColor('#333333')
LIGHT_GRAY = HexColor('#F5F5F5')

def create_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='CourseHeader',
        fontName='Helvetica',
        fontSize=9,
        textColor=UCLA_BLUE,
        alignment=TA_CENTER,
        spaceAfter=2
    ))

    styles.add(ParagraphStyle(
        name='CaseTitle',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=UCLA_DARK_BLUE,
        alignment=TA_CENTER,
        spaceAfter=4,
        spaceBefore=4
    ))

    styles.add(ParagraphStyle(
        name='WeekSubtitle',
        fontName='Helvetica',
        fontSize=10,
        textColor=DARK_GRAY,
        alignment=TA_CENTER,
        spaceAfter=12
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=UCLA_DARK_BLUE,
        spaceBefore=12,
        spaceAfter=6
    ))

    # Modify existing BodyText style
    styles['BodyText'].fontName = 'Helvetica'
    styles['BodyText'].fontSize = 10
    styles['BodyText'].textColor = DARK_GRAY
    styles['BodyText'].alignment = TA_JUSTIFY
    styles['BodyText'].spaceAfter = 8
    styles['BodyText'].leading = 14

    styles.add(ParagraphStyle(
        name='CaseSummary',
        fontName='Helvetica-Oblique',
        fontSize=10,
        textColor=DARK_GRAY,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14,
        leftIndent=10,
        rightIndent=10,
        borderPadding=5
    ))

    styles.add(ParagraphStyle(
        name='QuestionNumber',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=UCLA_BLUE,
        spaceBefore=8,
        spaceAfter=2
    ))

    styles.add(ParagraphStyle(
        name='QuestionText',
        fontName='Helvetica',
        fontSize=10,
        textColor=DARK_GRAY,
        leftIndent=20,
        spaceAfter=6,
        leading=13
    ))

    styles.add(ParagraphStyle(
        name='SourceText',
        fontName='Helvetica',
        fontSize=8,
        textColor=HexColor('#666666'),
        spaceBefore=12,
        leading=11
    ))

    styles.add(ParagraphStyle(
        name='InstructorHeader',
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=HexColor('#C62828'),
        spaceBefore=12,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name='TeachingNote',
        fontName='Helvetica',
        fontSize=9,
        textColor=DARK_GRAY,
        leftIndent=20,
        spaceAfter=4,
        leading=12
    ))

    styles.add(ParagraphStyle(
        name='KeyInsightBox',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=UCLA_DARK_BLUE,
        alignment=TA_CENTER,
        spaceBefore=10,
        spaceAfter=6,
        backColor=HexColor('#E3F2FD'),
        borderPadding=8
    ))

    return styles

def create_student_handout(filename, week_num, week_topic, case_title, case_summary, key_facts, questions, sources):
    """Create student-facing handout (no key insights)"""
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    styles = create_styles()
    story = []

    # Header
    story.append(Paragraph("MGMT298D: Science and Strategy of AI | UCLA Anderson", styles['CourseHeader']))
    story.append(Paragraph(case_title, styles['CaseTitle']))
    story.append(Paragraph(f"Week {week_num}: {week_topic}", styles['WeekSubtitle']))

    # Divider
    story.append(HRFlowable(width="100%", thickness=2, color=UCLA_GOLD, spaceAfter=10))

    # Case Summary
    story.append(Paragraph("The Situation", styles['SectionHeader']))
    story.append(Paragraph(case_summary, styles['CaseSummary']))

    # Key Facts Table
    story.append(Paragraph("Key Facts", styles['SectionHeader']))

    table_data = [[Paragraph(f"<b>{k}</b>", styles['BodyText']), Paragraph(v, styles['BodyText'])] for k, v in key_facts]

    table = Table(table_data, colWidths=[2.2*inch, 4.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
    ]))
    story.append(table)

    # Discussion Questions
    story.append(Paragraph("Discussion Questions", styles['SectionHeader']))

    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"Question {i}", styles['QuestionNumber']))
        story.append(Paragraph(q['question'], styles['QuestionText']))

    # Sources with URLs
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#CCCCCC'), spaceAfter=6))

    source_text = "<b>Sources:</b><br/>"
    for src in sources:
        if src.get('url'):
            source_text += f"• {src['name']}: <link href='{src['url']}'>{src['url']}</link><br/>"
        else:
            source_text += f"• {src['name']}<br/>"

    story.append(Paragraph(source_text, styles['SourceText']))

    doc.build(story)
    print(f"Created: {filename}")


def create_instructor_notes(filename, week_num, week_topic, case_title, questions, key_insight, teaching_tips):
    """Create instructor-only teaching notes"""
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    styles = create_styles()
    story = []

    # Header
    story.append(Paragraph("INSTRUCTOR NOTES - DO NOT DISTRIBUTE", styles['InstructorHeader']))
    story.append(Paragraph(case_title, styles['CaseTitle']))
    story.append(Paragraph(f"Week {week_num}: {week_topic}", styles['WeekSubtitle']))

    story.append(HRFlowable(width="100%", thickness=2, color=HexColor('#C62828'), spaceAfter=10))

    # Key Insight (what should emerge from discussion)
    story.append(Paragraph("Key Insight (Goal of Discussion)", styles['InstructorHeader']))
    story.append(Paragraph(key_insight, styles['KeyInsightBox']))

    # Teaching notes for each question
    story.append(Paragraph("Discussion Guide", styles['SectionHeader']))

    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"Question {i}: {q['question']}", styles['QuestionNumber']))
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Teaching Notes:</b>", styles['TeachingNote']))
        for note in q['notes']:
            story.append(Paragraph(f"• {note}", styles['TeachingNote']))
        story.append(Spacer(1, 6))

    # General teaching tips
    if teaching_tips:
        story.append(Paragraph("Additional Teaching Tips", styles['InstructorHeader']))
        for tip in teaching_tips:
            story.append(Paragraph(f"• {tip}", styles['TeachingNote']))

    doc.build(story)
    print(f"Created: {filename}")


# ============================================
# CASE DATA
# ============================================

cases = [
    # WEEK 1
    {
        "week": 1,
        "topic": "Linear Regression",
        "id": "Week1A_Zillow",
        "title": "Zillow's $500M iBuying Failure",
        "summary": "In 2021, Zillow shut down its home-flipping business after losing over $500 million in a single quarter. The company laid off 25% of its workforce. Here's the twist: Zillow's Zestimate algorithm had been running successfully for 15 years, providing home value estimates to millions of users. When they decided to actually buy homes based on these predictions, everything fell apart within two years. CEO Rich Barton admitted: 'We've determined the unpredictability in forecasting home prices far exceeds what we anticipated.'",
        "key_facts": [
            ("Zestimate launched", "2006 (15 years before iBuying)"),
            ("iBuying launched", "2018"),
            ("Homes purchased (2021)", "27,000+"),
            ("Q3 2021 loss", "$500M+"),
            ("Employees laid off", "2,000 (25% of workforce)"),
            ("Root cause", "Systematic overpayment during volatile market"),
        ],
        "questions": [
            {
                "question": "Zillow's Zestimate had been running for 15 years before iBuying. What fundamentally changed when they moved from providing estimates to acting on them?",
                "notes": [
                    "Providing information vs. making decisions have different error tolerances",
                    "When you're just informing, being 5% off is fine—users adjust",
                    "When you're betting $500K per house, 5% off is catastrophic",
                    "The model didn't get worse; the stakes changed"
                ]
            },
            {
                "question": "What features would you include in a home price prediction model? Which would be hardest to capture in data?",
                "notes": [
                    "Easy: Square footage, bedrooms, location, lot size, age",
                    "Medium: School quality, crime rates, commute times",
                    "Hard: Neighborhood 'feel,' future development plans, market momentum",
                    "Impossible: What a specific buyer will pay (idiosyncratic preferences)"
                ]
            },
            {
                "question": "Zillow cited 'labor and supply shortages' as factors. How might a regression model trained on historical data fail during rapidly changing market conditions?",
                "notes": [
                    "Training data was pre-pandemic 'normal' market",
                    "2021 market was unprecedented: supply shortages, migration patterns, interest rates",
                    "Models assume the future resembles the past (stationarity assumption)",
                    "Feature engineering can't capture regime changes"
                ]
            },
            {
                "question": "If you were the CEO, what guardrails would you have put in place before betting $1 billion on algorithmic home purchases?",
                "notes": [
                    "Human review for high-value purchases",
                    "Maximum bid limits (cap the downside)",
                    "Geographic diversification requirements",
                    "Real-time model monitoring for drift",
                    "Stress testing against historical volatility"
                ]
            }
        ],
        "key_insight": "The gap between 'good at prediction' and 'good enough to bet the company' is enormous. Prediction accuracy isn't the same as decision quality.",
        "sources": [
            {"name": "WSJ: Zillow Quits Home-Flipping Business", "url": "https://www.wsj.com/articles/zillow-quits-home-flipping-business-cites-inability-to-forecast-prices-11635883500"},
            {"name": "The Verge: Zillow's iBuying Debacle", "url": "https://www.theverge.com/2021/11/1/22758176/zillow-offers-ibuying-housing-market-inventory-homes"},
            {"name": "Bloomberg: Inside Zillow's Plan to Buy 5,000 Homes a Month", "url": "https://www.bloomberg.com/news/articles/2021-11-02/zillow-s-zestimate-proves-hard-to-trust-in-home-flipping-debacle"},
        ],
        "teaching_tips": [
            "Start by asking who has used Zestimate—most students will have",
            "Emphasize this wasn't a 'bad model'—it was a misapplication of a good model",
            "Connect to the lecture: What MAE would have been acceptable for iBuying?",
            "This case sets up the theme: prediction ≠ decision quality"
        ]
    },
    {
        "week": 1,
        "topic": "Linear Regression",
        "id": "Week1B_Moneyball",
        "title": "Moneyball: How the Oakland A's Beat the Yankees with Data",
        "summary": "In 2002, the Oakland A's had the third-lowest payroll in baseball—$41 million compared to the Yankees' $125 million. Despite this, they won 103 games and made the playoffs. GM Billy Beane's secret? He used regression analysis to discover that the market was mispricing players. Traditional scouts valued batting average, stolen bases, and RBIs. Beane's analysis showed that on-base percentage and slugging percentage were better predictors of wins—and players with those skills were undervalued. By 2016, all 30 NBA teams and 26 of 30 MLB teams had dedicated analytics departments.",
        "key_facts": [
            ("A's payroll (2002)", "$41M (3rd lowest in MLB)"),
            ("Yankees payroll (2002)", "$125M (highest)"),
            ("A's wins (2002)", "103 games, made playoffs"),
            ("Key discovery", "On-base percentage was undervalued"),
            ("By 2016", "All 30 NBA teams had analytics staff"),
            ("Outcome", "Market inefficiency was competed away"),
        ],
        "questions": [
            {
                "question": "What made on-base percentage 'undervalued' compared to batting average? Why did traditional scouts miss this?",
                "notes": [
                    "Batting average was traditional, visible, discussed by media",
                    "OBP included walks, which seemed 'passive' to scouts",
                    "Regression revealed OBP correlated more strongly with runs scored",
                    "Market inefficiency: scouts used intuition, not data"
                ]
            },
            {
                "question": "The A's advantage eroded as other teams adopted analytics. How do you maintain competitive advantage when your methods can be copied?",
                "notes": [
                    "First-mover advantage is temporary",
                    "Must continuously innovate (new data sources, better models)",
                    "Today's edge: proprietary data, faster iteration, better execution",
                    "Similar pattern in business: early AI adopters lose edge as tools commoditize"
                ]
            },
            {
                "question": "Baseball is data-rich with clear outcomes. What business domains have similar characteristics? Which don't?",
                "notes": [
                    "Similar: Finance (stock prices), retail (sales), digital marketing (clicks)",
                    "Different: B2B sales (few observations), healthcare (long feedback loops), creativity (subjective outcomes)",
                    "Key requirements: abundant data, clear outcome variable, fast feedback"
                ]
            },
            {
                "question": "What are the limits of purely data-driven decision making in sports or business?",
                "notes": [
                    "Works when outcomes are measurable and data is plentiful",
                    "Struggles with: team chemistry, leadership, adaptability",
                    "Can create blind spots (what the data doesn't capture)",
                    "Human judgment still matters for strategy, culture, rare events"
                ]
            }
        ],
        "key_insight": "Regression reveals which variables actually matter—often not the ones intuition suggests. But analytical advantages erode as competitors adopt similar methods.",
        "sources": [
            {"name": "Book: Moneyball by Michael Lewis", "url": "https://www.amazon.com/Moneyball-Art-Winning-Unfair-Game/dp/0393324818"},
            {"name": "INFORMS: Analytics in Sports", "url": "https://www.informs.org/Impact/O.R.-Analytics-Success-Stories/Sports"},
            {"name": "MIT Sloan Sports Analytics Conference", "url": "https://www.sloansportsconference.com/"},
            {"name": "FiveThirtyEight: The Complete History of MLB Analytics", "url": "https://fivethirtyeight.com/features/dont-be-fooled-by-baseballs-small-budget-success-stories/"},
        ],
        "teaching_tips": [
            "Ask how many students have seen the movie—use it as a touchpoint",
            "Emphasize this is about regression coefficients revealing truth",
            "Connect to lecture: This is what β coefficients tell us",
            "Good discussion on competitive advantage and data moats"
        ]
    },
    {
        "week": 1,
        "topic": "Linear Regression",
        "id": "Week1C_Airbnb",
        "title": "Airbnb Smart Pricing",
        "summary": "Airbnb hosts face a challenging pricing decision every day: price too high and the listing sits empty; price too low and you leave money on the table. Airbnb's Smart Pricing tool automatically adjusts prices based on 70+ factors: demand patterns, seasonality, local events, day of week, competitor rates, and listing characteristics. Their hybrid approach—combining structural economic models with machine learning—cut prediction errors in half compared to pure ML approaches. But here's the tension: Airbnb benefits when total bookings increase. Individual hosts benefit when their revenue increases. Are these goals always aligned?",
        "key_facts": [
            ("Factors considered", "70+ variables"),
            ("Approach", "Hybrid structural + ML model"),
            ("Result", "50% reduction in prediction error"),
            ("Challenge", "Many hosts don't trust algorithmic pricing"),
            ("Conflict", "Airbnb optimizes marketplace; hosts optimize individual revenue"),
            ("Data sources", "Events, weather, competitors, seasonality"),
        ],
        "questions": [
            {
                "question": "Why might hosts be reluctant to trust an algorithm with their pricing? How would you address this skepticism?",
                "notes": [
                    "Loss of control over a key business decision",
                    "Can't understand why the algorithm chose a price",
                    "Fear of racing to the bottom",
                    "Solution: Transparency, ability to set floors/ceilings, show reasoning"
                ]
            },
            {
                "question": "Airbnb benefits when hosts price optimally for the marketplace. What conflicts of interest might arise between Airbnb and individual hosts?",
                "notes": [
                    "Airbnb wants bookings (any booking generates fees)",
                    "Hosts want revenue (price × occupancy)",
                    "Algorithm might suggest lower prices than hosts would choose",
                    "Platform vs. participant incentive misalignment"
                ]
            },
            {
                "question": "The model incorporates local events like concerts and conferences. What data sources would you need? What events might be hard to predict?",
                "notes": [
                    "Easy: Scheduled events (concerts, sports, conferences)",
                    "Medium: Weather forecasts, holiday calendars",
                    "Hard: Breaking news, celebrity visits, viral moments",
                    "Integration challenge: Combining dozens of data feeds in real-time"
                ]
            },
            {
                "question": "How should Smart Pricing handle unprecedented situations—like a pandemic that eliminates travel demand?",
                "notes": [
                    "Historical data becomes irrelevant",
                    "Model will be wrong in unpredictable ways",
                    "Need: Human override capability, anomaly detection, rapid retraining",
                    "Similar challenge to Zillow: models assume stationarity"
                ]
            }
        ],
        "key_insight": "Pricing optimization is a natural application for regression—clear outcome variable, abundant data, actionable predictions. But platform incentives may not align with user incentives.",
        "sources": [
            {"name": "Airbnb Engineering: Aerosolve ML Framework", "url": "https://medium.com/airbnb-engineering/aerosolve-machine-learning-for-humans-55efcf602665"},
            {"name": "Airbnb: How Machine Learning Powers Pricing", "url": "https://medium.com/airbnb-engineering/learning-market-dynamics-for-optimal-pricing-97cffbcc53e3"},
            {"name": "Harvard Business Review: The Operational Transparency Case", "url": "https://hbr.org/2019/03/operational-transparency"},
        ],
        "teaching_tips": [
            "Good for discussing platform economics and incentive alignment",
            "Ask students: Would you use Smart Pricing as a host? Why or why not?",
            "Connect to regression: What's the target variable? (Revenue? Bookings? Occupancy?)",
            "Ties nicely to the pandemic disruption theme from Zillow"
        ]
    },
    # WEEK 2
    {
        "week": 2,
        "topic": "Tree-Based Predictions",
        "id": "Week2A_CapitalOne",
        "title": "Capital One: From Credit Cards to AI Pioneer",
        "summary": "Capital One was founded in 1988 on a radical premise: use data and analytics to make better credit decisions than competitors. Today, they use XGBoost and ensemble tree methods for fraud detection (preventing $1.5B in losses annually), credit risk assessment, and anti-money laundering. But here's the regulatory reality: credit decisions must be explainable. When you deny someone a loan, you must tell them why. This constraint shapes which models Capital One can deploy—and tree-based methods often win because you can trace the decision path.",
        "key_facts": [
            ("Founded", "1988 (data-first from day one)"),
            ("Fraud losses prevented", "$1.5B annually"),
            ("Model preference", "Tree-based (XGBoost) for explainability"),
            ("Regulatory requirement", "Must explain credit denials"),
            ("Key applications", "Fraud, credit risk, AML"),
            ("Competitive advantage", "Better models = better margins"),
        ],
        "questions": [
            {
                "question": "Capital One was founded on the premise that better models = better credit decisions. What are the competitive advantages of being a 'data-first' financial institution?",
                "notes": [
                    "Can price risk more accurately (charge appropriate rates)",
                    "Approve customers competitors would reject (expand market)",
                    "Detect fraud faster (reduce losses)",
                    "Continuously improve with more data (flywheel effect)"
                ]
            },
            {
                "question": "Credit decisions must be explainable (regulatory requirement). How does this constraint affect model choice? Why might tree-based models be preferred over neural networks?",
                "notes": [
                    "Trees have clear decision paths: 'You were denied because income < $X and debt > $Y'",
                    "Neural networks are black boxes—hard to explain specific decisions",
                    "Regulation (ECOA, FCRA) requires adverse action notices",
                    "Trade-off: Some accuracy for interpretability"
                ]
            },
            {
                "question": "ML models for credit scoring can inadvertently encode historical biases. How should a company balance predictive accuracy with fairness?",
                "notes": [
                    "Historical data reflects historical discrimination",
                    "Proxies for protected classes (zip code → race)",
                    "Fairness metrics: demographic parity, equalized odds",
                    "Tension: Removing predictive features may reduce accuracy"
                ]
            },
            {
                "question": "Capital One moved from rules-based fraud detection to ML-based. What are the trade-offs of each approach?",
                "notes": [
                    "Rules: Transparent, predictable, but rigid and easily gamed",
                    "ML: Adaptive, catches complex patterns, but less transparent",
                    "Fraud evolves—rules can't keep up",
                    "Best approach often combines both"
                ]
            }
        ],
        "key_insight": "Tree models remain industry workhorses because they balance accuracy with interpretability. In regulated industries, being able to explain your decision is as important as making the right one.",
        "sources": [
            {"name": "Capital One Tech Blog: Machine Learning at Scale", "url": "https://www.capitalone.com/tech/machine-learning/"},
            {"name": "Harvard Business School Case: Capital One", "url": "https://www.hbs.edu/faculty/Pages/item.aspx?num=26621"},
            {"name": "Federal Reserve: Model Risk Management Guidance", "url": "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm"},
            {"name": "Consumer Financial Protection Bureau: Adverse Action Notices", "url": "https://www.consumerfinance.gov/"},
        ],
        "teaching_tips": [
            "Great case for discussing the accuracy-interpretability trade-off",
            "Ask: Would you want a black-box model deciding your loan application?",
            "Connect to lecture: Feature importance in trees = explainability",
            "Fairness discussion can be extended or shortened based on time"
        ]
    },
    {
        "week": 2,
        "topic": "Tree-Based Predictions",
        "id": "Week2B_JPMorgan",
        "title": "JPMorgan Chase: AI in Banking at Scale",
        "summary": "JPMorgan Chase has 450+ AI use cases in development and a $17 billion annual technology budget. Their AI-driven fraud detection prevents $1.5B in losses annually with 98% accuracy. Their COIN (Contract Intelligence) system uses ML to review commercial loan agreements, reducing review time from 360,000 hours annually to seconds. But scale brings challenges: with 450+ use cases, how do you prioritize? And when your fraud model is 98% accurate, what's the cost of the 2% errors?",
        "key_facts": [
            ("AI use cases", "450+ in development"),
            ("Technology budget", "$17B annually"),
            ("Fraud detection accuracy", "98%"),
            ("Fraud losses prevented", "$1.5B annually"),
            ("COIN time savings", "360,000 hours → seconds"),
            ("COIN application", "Commercial loan agreement review"),
        ],
        "questions": [
            {
                "question": "JPMorgan has 450+ AI use cases. How should a large organization prioritize which ML projects to pursue?",
                "notes": [
                    "ROI calculation: value created vs. development cost",
                    "Strategic importance: competitive advantage vs. table stakes",
                    "Data availability and quality",
                    "Regulatory and risk considerations",
                    "Build vs. buy decisions"
                ]
            },
            {
                "question": "Their fraud detection achieves 98% accuracy. What's the cost of the 2% errors? Are false positives or false negatives worse?",
                "notes": [
                    "False negative: Fraud gets through → direct financial loss",
                    "False positive: Legitimate transaction blocked → customer frustration, lost business",
                    "Context matters: $10 transaction vs. $10,000 transaction",
                    "Banks typically prefer false positives (can verify with customer)"
                ]
            },
            {
                "question": "COIN reduces loan review from 360,000 hours annually to seconds. What happens to the employees who previously did this work?",
                "notes": [
                    "Retraining for higher-value tasks",
                    "Some displacement is inevitable",
                    "Shift from execution to oversight and exception handling",
                    "New roles: AI trainers, model monitors, exception handlers"
                ]
            },
            {
                "question": "As a regulated industry, banking faces unique AI constraints. How do these shape which models can be deployed?",
                "notes": [
                    "Model risk management requirements (SR 11-7)",
                    "Fair lending laws limit certain variables",
                    "Audit and documentation requirements",
                    "Often favors simpler, more interpretable models"
                ]
            }
        ],
        "key_insight": "Enterprise AI requires balancing accuracy, explainability, and regulatory compliance. Scale brings prioritization challenges—not every good idea is worth pursuing.",
        "sources": [
            {"name": "JPMorgan Annual Report: Technology Section", "url": "https://www.jpmorganchase.com/ir/annual-report"},
            {"name": "JPMorgan: Machine Learning and AI", "url": "https://www.jpmorgan.com/technology/artificial-intelligence"},
            {"name": "American Banker: JPMorgan's AI Strategy", "url": "https://www.americanbanker.com/"},
            {"name": "Reuters: COIN Contract Intelligence", "url": "https://www.reuters.com/article/us-jpmorgan-tech-loan-idUSKBN17Y0JK"},
        ],
        "teaching_tips": [
            "Good for discussing enterprise AI strategy and prioritization",
            "The 98% accuracy question is excellent for precision/recall discussion",
            "COIN connects well to Week 7 (NLP/Transformers)",
            "Ask: If you were CTO, which of 450 projects would you prioritize?"
        ]
    },
    {
        "week": 2,
        "topic": "Tree-Based Predictions",
        "id": "Week2C_Healthcare",
        "title": "Healthcare AI: FDA-Approved Devices",
        "summary": "Over 924 AI medical devices have received FDA clearance, with 75%+ focused on radiology. Viz.ai's stroke detection tool is deployed in 1,600+ hospitals, reducing treatment time by an average of one hour—critical when 1.9 million neurons die every minute during a stroke. However, only 5% of approved devices underwent prospective clinical testing before approval. The FDA's 510(k) pathway allows devices to be approved based on similarity to existing products, raising questions about validation rigor.",
        "key_facts": [
            ("FDA-cleared AI devices", "924+"),
            ("Radiology focus", "75%+ of approvals"),
            ("Viz.ai deployment", "1,600+ hospitals"),
            ("Time savings (Viz.ai)", "1 hour average reduction"),
            ("Stroke impact", "1.9M neurons die per minute"),
            ("Prospective testing", "Only 5% of devices"),
        ],
        "questions": [
            {
                "question": "Why has radiology been the leading domain for FDA-approved AI? What characteristics make it particularly suitable?",
                "notes": [
                    "Clear ground truth (biopsy results, follow-up imaging)",
                    "Abundant digital data (images already stored)",
                    "Defined tasks (detect tumor, measure size)",
                    "Workflow integration is straightforward",
                    "Lower direct patient contact reduces some risks"
                ]
            },
            {
                "question": "Only 5% of approved devices underwent prospective clinical testing. What are the risks of this approach?",
                "notes": [
                    "510(k) pathway: 'substantially equivalent' to existing device",
                    "Training data may not match deployment population",
                    "Edge cases and rare conditions may not be covered",
                    "Post-market surveillance becomes critical"
                ]
            },
            {
                "question": "Viz.ai reduces stroke treatment time by one hour. How would you calculate the economic and health value of this improvement?",
                "notes": [
                    "1.9M neurons/minute → 114M neurons saved per patient",
                    "Better outcomes reduce long-term care costs",
                    "Quality-adjusted life years (QALYs) saved",
                    "Compare device cost to treatment cost savings"
                ]
            },
            {
                "question": "What's the appropriate role for AI in medical diagnosis—replacement, augmentation, or second opinion?",
                "notes": [
                    "Current FDA stance: AI as 'decision support'",
                    "Liability questions: Who's responsible for errors?",
                    "Radiologist + AI often outperforms either alone",
                    "Risk tolerance varies by condition severity"
                ]
            }
        ],
        "key_insight": "Classification models in healthcare must balance accuracy with safety and regulatory requirements. The 510(k) pathway enables faster deployment but may not catch all failure modes.",
        "sources": [
            {"name": "FDA: Artificial Intelligence and Machine Learning in Medical Devices", "url": "https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices"},
            {"name": "JAMA: FDA Regulation of AI Medical Devices", "url": "https://jamanetwork.com/journals/jama/article-abstract/2799321"},
            {"name": "Viz.ai: Company Overview", "url": "https://www.viz.ai/"},
            {"name": "Nature Medicine: AI in Radiology", "url": "https://www.nature.com/articles/s41591-018-0307-0"},
        ],
        "teaching_tips": [
            "Great case for discussing classification metrics in high-stakes settings",
            "Ask: What false positive rate is acceptable for cancer screening?",
            "Connect to lecture: Precision vs. recall trade-off",
            "Can extend to discuss algorithmic bias in healthcare"
        ]
    },
    # WEEK 3
    {
        "week": 3,
        "topic": "Clustering & Collaborative Filtering",
        "id": "Week3A_Spotify",
        "title": "Spotify & Netflix: Taste Communities",
        "summary": "Netflix has 1,300+ 'taste communities' that cut across geography and demographics—a teenager in Brazil and a retiree in Japan might be in the same cluster based on viewing behavior. Traditional market research segments by age, income, and location. Netflix discovered these variables are poor predictors of taste. Spotify takes a similar approach, training collaborative filtering on 700 million user-generated playlists. The insight: behavior reveals preferences better than demographics.",
        "key_facts": [
            ("Netflix taste communities", "1,300+"),
            ("Spotify training data", "700M user playlists"),
            ("Key insight", "Behavior > demographics for taste"),
            ("Cluster example", "Teen in Brazil = Retiree in Japan"),
            ("Traditional segmentation", "Age, income, location (poor predictors)"),
            ("Business impact", "Content investment, personalization"),
        ],
        "questions": [
            {
                "question": "Traditional market research segments by demographics. Why did Netflix and Spotify abandon this approach?",
                "notes": [
                    "Demographics assume preferences are determined by who you are",
                    "Behavior reveals what you actually like",
                    "Two 25-year-olds may have completely different tastes",
                    "Demographics are proxies; behavior is direct measurement"
                ]
            },
            {
                "question": "Netflix found that a teenager in Brazil and a retiree in Japan might be in the same 'taste cluster.' What are the implications for content creation?",
                "notes": [
                    "Global content can find global audiences",
                    "Don't need to create 'content for seniors' or 'content for teens'",
                    "Niche content becomes viable (long tail)",
                    "Reduces reliance on demographic assumptions"
                ]
            },
            {
                "question": "Spotify uses clustering for both users AND songs. How might clustering songs differ from clustering users?",
                "notes": [
                    "Song features: tempo, key, energy, acousticness",
                    "User features: listening history, playlist behavior",
                    "Songs can be in multiple clusters (a song is both 'workout' and 'pop')",
                    "User clusters are more fluid over time"
                ]
            },
            {
                "question": "If you were launching a new product, how would you balance demographic segmentation with behavioral clustering?",
                "notes": [
                    "Start with demographics (you have no behavior data)",
                    "Shift to behavioral as you collect data",
                    "Cold start problem: new users have no history",
                    "Hybrid approaches often work best"
                ]
            }
        ],
        "key_insight": "Unsupervised learning reveals structure humans wouldn't think to look for. Behavior-based segments often outperform demographic assumptions.",
        "sources": [
            {"name": "Netflix Tech Blog: Learning a Personalized Homepage", "url": "https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a"},
            {"name": "Spotify Engineering: Discover Weekly", "url": "https://engineering.atspotify.com/2015/09/everynoise-at-once/"},
            {"name": "Quartz: Netflix's Taste Communities", "url": "https://qz.com/1187837/netflix-has-identified-more-than-1300-taste-communities-around-the-world"},
            {"name": "Harvard Business Review: How Netflix Uses Analytics", "url": "https://hbr.org/2018/01/how-netflix-uses-data-science-to-drive-success"},
        ],
        "teaching_tips": [
            "Ask students what demographic segment they're in vs. their actual Netflix taste",
            "Good discussion on cold start problem",
            "Connect to lecture: K-means finding structure without labels",
            "Can discuss filter bubbles as a downside"
        ]
    },
    {
        "week": 3,
        "topic": "Clustering & Collaborative Filtering",
        "id": "Week3B_Starbucks",
        "title": "Starbucks Deep Brew: Personalization at Scale",
        "summary": "Starbucks' Deep Brew platform analyzes purchase history, store locations, time of day, weather, and individual preferences to personalize offers for millions of customers. BOGO (Buy One Get One) offers see 45% completion among new customers. Their 6-year average member tenure enables rich behavioral segmentation—they know your order before you do. The challenge: balancing personalization with privacy, and driving traffic without training customers to only buy on discount.",
        "key_facts": [
            ("Platform", "Deep Brew"),
            ("BOGO completion rate", "45% (new customers)"),
            ("Average member tenure", "6 years"),
            ("Variables analyzed", "Purchases, location, time, weather"),
            ("Personalization challenge", "Avoid training discount-only behavior"),
            ("Scale", "Millions of daily transactions"),
        ],
        "questions": [
            {
                "question": "Starbucks has data on millions of daily transactions. What customer segments would you expect to emerge from this data?",
                "notes": [
                    "Morning commuters (same order, same time, same store)",
                    "Weekend explorers (different drinks, different stores)",
                    "Deal seekers (respond to offers, price sensitive)",
                    "Loyalists (high frequency, low price sensitivity)",
                    "Occasional visitors (holidays, travel)"
                ]
            },
            {
                "question": "They found that Discount offers have higher completion rates than BOGO. Why might this be, and how would you test it?",
                "notes": [
                    "Discount is simpler (immediate value)",
                    "BOGO requires finding a second person or second purchase",
                    "A/B test different offer types on similar segments",
                    "Lifetime value matters more than single offer completion"
                ]
            },
            {
                "question": "Deep Brew considers weather in its recommendations. What other contextual factors might influence coffee purchases?",
                "notes": [
                    "Time of day (morning vs. afternoon)",
                    "Day of week (workday vs. weekend)",
                    "Nearby events (sports games, concerts)",
                    "Season (iced drinks in summer)",
                    "Even current wait time at store"
                ]
            },
            {
                "question": "How might Starbucks' segmentation strategy differ from a grocery store's approach?",
                "notes": [
                    "Starbucks: High frequency, low basket size",
                    "Grocery: Lower frequency, large basket size",
                    "Starbucks can personalize specific products",
                    "Grocery must recommend across categories",
                    "Both can use purchase timing and patterns"
                ]
            }
        ],
        "key_insight": "Clustering enables personalization at scale—treating millions of customers as individuals. The goal is finding actionable segments that drive behavior.",
        "sources": [
            {"name": "Starbucks: Deep Brew AI Platform", "url": "https://stories.starbucks.com/stories/2019/starbucks-unveils-new-deep-brew-ai-platform/"},
            {"name": "Harvard Business Review: Starbucks Reinvention", "url": "https://hbr.org/2022/11/starbucks-new-ceo-shares-his-strategy-for-a-turnaround"},
            {"name": "Forbes: How Starbucks Uses AI", "url": "https://www.forbes.com/sites/bernardmarr/2018/05/28/starbucks-using-big-data-analytics-and-artificial-intelligence-to-boost-performance/"},
        ],
        "teaching_tips": [
            "Ask who has the Starbucks app and what offers they get",
            "Good for discussing the discount trap (training customers to wait for deals)",
            "Connect to lecture: Clustering to find actionable segments",
            "Can discuss privacy concerns with location tracking"
        ]
    },
    {
        "week": 3,
        "topic": "Clustering & Collaborative Filtering",
        "id": "Week3C_Healthcare",
        "title": "Healthcare: High-Cost Patient Segmentation",
        "summary": "5% of patients account for 50% of healthcare spending. Researchers applied clustering algorithms to segment top-decile Medicare spending patients into clinically meaningful subgroups. The insight: 'high-cost patients' aren't a monolithic group. Some have chronic conditions requiring ongoing management; others have acute events requiring intensive short-term care. One-size-fits-all interventions waste resources. Targeted interventions based on cluster membership can improve outcomes and reduce costs.",
        "key_facts": [
            ("Spending concentration", "5% of patients = 50% of costs"),
            ("Target population", "Top-decile Medicare spenders"),
            ("Problem with averages", "'High-cost' is not one group"),
            ("Subgroup examples", "Chronic management vs. acute events"),
            ("Intervention approach", "Targeted by cluster membership"),
            ("Goal", "Better outcomes at lower cost"),
        ],
        "questions": [
            {
                "question": "Why might clustering be more useful than rules for identifying patient subgroups?",
                "notes": [
                    "Rules require knowing the categories in advance",
                    "Clustering discovers natural groupings in data",
                    "Patients don't fit neat boxes (multiple conditions)",
                    "Clusters can reveal unexpected combinations"
                ]
            },
            {
                "question": "High-cost patients aren't a monolithic group. What distinct subgroups might exist within this population?",
                "notes": [
                    "Chronic disease managers (diabetes, heart failure)",
                    "Acute crisis responders (ER frequent flyers)",
                    "End-of-life care recipients",
                    "Mental health + physical health combination",
                    "Post-surgical recovery"
                ]
            },
            {
                "question": "Once you identify subgroups, how would you design different interventions for each?",
                "notes": [
                    "Chronic: Care coordination, medication management",
                    "Acute: Social work, housing support, mental health",
                    "End-of-life: Palliative care, advance directives",
                    "One-size-fits-all wastes resources"
                ]
            },
            {
                "question": "What are the ethical considerations of targeting interventions based on predicted cost?",
                "notes": [
                    "Are we helping patients or reducing costs?",
                    "Risk of denying care to those predicted expensive",
                    "Equity concerns: Who ends up in which cluster?",
                    "Need patient-centered goals, not just cost reduction"
                ]
            }
        ],
        "key_insight": "Clustering can reveal actionable subgroups that enable targeted intervention. In healthcare, treating everyone the same often means treating everyone poorly.",
        "sources": [
            {"name": "JGIM: Clustering High-Cost Patients", "url": "https://link.springer.com/article/10.1007/s11606-019-05336-7"},
            {"name": "Health Affairs: The 5% Who Account for Half of Spending", "url": "https://www.healthaffairs.org/doi/10.1377/hlthaff.2012.1277"},
            {"name": "NEJM Catalyst: Segmenting High-Need Patients", "url": "https://catalyst.nejm.org/doi/full/10.1056/CAT.17.0435"},
        ],
        "teaching_tips": [
            "Good case for discussing unsupervised learning in high-stakes settings",
            "Ethical discussion can be as long or short as desired",
            "Connect to lecture: Clustering reveals structure without labels",
            "Ask: How would you validate that clusters are clinically meaningful?"
        ]
    },
    # WEEK 4
    {
        "week": 4,
        "topic": "Reinforcement Learning",
        "id": "Week4A_Uber",
        "title": "Uber: Marketplace Optimization at Scale",
        "summary": "Uber's reinforcement learning system runs in 400+ cities—the largest production RL deployment for ridesharing. A naive algorithm would match each rider to the nearest driver (greedy matching). But this creates problems: if you send all drivers to the airport, downtown gets starved. Uber's RL optimizes for the whole marketplace, sometimes sending a driver to pick up a slightly farther rider to maintain balance across the city. The system learns which actions lead to better long-term outcomes.",
        "key_facts": [
            ("Cities with RL", "400+"),
            ("Scale", "Largest production RL for ridesharing"),
            ("Naive approach", "Greedy matching (nearest driver)"),
            ("RL approach", "Optimize whole marketplace balance"),
            ("Key insight", "Local optimization hurts global outcomes"),
            ("Learning signal", "Wait times, completion rates, balance"),
        ],
        "questions": [
            {
                "question": "A greedy algorithm would assign the nearest driver to each rider. Why might this be suboptimal for the overall marketplace?",
                "notes": [
                    "Creates supply deserts in some areas",
                    "All drivers rush to high-demand areas (airport)",
                    "Longer wait times on average even if some are short",
                    "Doesn't consider future demand"
                ]
            },
            {
                "question": "Uber uses RL to balance supply and demand across a city. What happens if RL optimizes too aggressively for one metric?",
                "notes": [
                    "Optimize only wait time: Drivers hate it (short trips)",
                    "Optimize only driver earnings: Riders wait too long",
                    "Need multi-objective optimization",
                    "Goodhart's Law: When a measure becomes a target..."
                ]
            },
            {
                "question": "Surge pricing is controversial but Uber argues it's necessary. How would you explain the economic rationale to a skeptical customer?",
                "notes": [
                    "Higher prices reduce demand (fewer riders)",
                    "Higher prices increase supply (more drivers come online)",
                    "Without surge, demand > supply = no cars available",
                    "Surge is allocation mechanism, not just profit"
                ]
            },
            {
                "question": "DoorDash and Lyft face similar challenges. What transfers to food delivery? What's different?",
                "notes": [
                    "Similar: Balancing supply/demand, batching orders",
                    "Different: Food has freshness constraints (time limit)",
                    "Different: Restaurants have limited capacity",
                    "Different: Multi-pickup routes are more common"
                ]
            }
        ],
        "key_insight": "RL shines in sequential decisions where actions affect future states. Greedy optimization often fails when local decisions have global consequences.",
        "sources": [
            {"name": "Uber Engineering: Reinforcement Learning for Marketplace", "url": "https://eng.uber.com/reinforcement-learning-market-optimization/"},
            {"name": "Uber AI: Making Uber's Marketplace Real-Time", "url": "https://www.uber.com/blog/marketplace-optimization/"},
            {"name": "MIT Technology Review: How Uber Uses AI", "url": "https://www.technologyreview.com/2018/08/21/140356/how-uber-uses-machine-learning-to-match-drivers-and-customers/"},
        ],
        "teaching_tips": [
            "Most students use rideshare—ask about surge pricing experiences",
            "Good for explaining the explore/exploit trade-off",
            "Connect to lecture: States, actions, rewards framework",
            "Can discuss ethical concerns about algorithmic pricing"
        ]
    },
    {
        "week": 4,
        "topic": "Reinforcement Learning",
        "id": "Week4B_AlphaGo",
        "title": "AlphaGo: The Game That Changed AI",
        "summary": "In March 2016, AlphaGo defeated world champion Lee Sedol 4-1 in a match watched by 200 million people. Go has more possible board positions than atoms in the universe—brute force search is impossible. AlphaGo combined deep learning with reinforcement learning, training on millions of human games then playing millions of games against itself. AlphaGo Zero went further: starting from zero human knowledge, it learned entirely through self-play and defeated the original AlphaGo 100-0.",
        "key_facts": [
            ("Match result", "AlphaGo 4, Lee Sedol 1"),
            ("Viewers", "200 million"),
            ("Go complexity", "More positions than atoms in universe"),
            ("AlphaGo training", "Human games + self-play"),
            ("AlphaGo Zero", "No human data, pure self-play"),
            ("Zero vs Original", "100-0"),
        ],
        "questions": [
            {
                "question": "Go has more possible positions than atoms in the universe. Why did reinforcement learning succeed where brute-force search couldn't?",
                "notes": [
                    "Can't enumerate all positions (unlike chess)",
                    "RL learns value function: Which positions are good?",
                    "Doesn't need to search everything—just promising paths",
                    "Neural network generalizes from seen positions to unseen"
                ]
            },
            {
                "question": "AlphaGo Zero learned without human data and discovered unconventional strategies. What does this suggest about human expertise?",
                "notes": [
                    "Humans may have suboptimal conventional wisdom",
                    "Self-play can discover strategies humans never considered",
                    "But: Humans learn with far less data",
                    "Human expertise is sample-efficient; AI is not"
                ]
            },
            {
                "question": "AlphaGo's techniques have been applied to protein folding (AlphaFold) and chip design. What makes a problem suitable for this approach?",
                "notes": [
                    "Clear objective function (win/lose, structure fits)",
                    "Ability to simulate/evaluate outcomes",
                    "Large but structured search space",
                    "Not suitable: Open-ended problems, unclear goals"
                ]
            },
            {
                "question": "What's the business value of solving games? Why did DeepMind invest so heavily in Go?",
                "notes": [
                    "Games as benchmark for AI progress",
                    "Techniques transfer to real problems",
                    "PR value and talent attraction",
                    "Google acquired DeepMind before AlphaGo"
                ]
            }
        ],
        "key_insight": "RL can discover strategies that surpass human expertise when rules are clear and simulation is cheap. But the techniques require careful adaptation for real-world problems.",
        "sources": [
            {"name": "DeepMind: AlphaGo", "url": "https://deepmind.google/technologies/alphago/"},
            {"name": "Nature: Mastering the Game of Go", "url": "https://www.nature.com/articles/nature16961"},
            {"name": "Nature: Mastering Go without Human Knowledge", "url": "https://www.nature.com/articles/nature24270"},
            {"name": "Wired: The Mystery of Go", "url": "https://www.wired.com/2016/05/google-alpha-go-ai/"},
        ],
        "teaching_tips": [
            "Show the Move 37 video clip if time permits",
            "AlphaGo Zero is the more important result pedagogically",
            "Connect to lecture: Self-play as RL without human labels",
            "Good transition to discussing AI capabilities and limitations"
        ]
    },
    {
        "week": 4,
        "topic": "Reinforcement Learning",
        "id": "Week4C_Amazon",
        "title": "Amazon: Deep RL for Inventory Management",
        "summary": "Amazon uses reinforcement learning for inventory optimization, handling stochastic lead times and correlated demand across millions of products. During Prime Day 2023, AI systems processed tens of millions of goods movements daily without widespread stockouts. The challenge: inventory decisions today affect availability tomorrow. Order too much and capital is tied up in warehouses; order too little and customers see 'out of stock.' RL learns the long-term consequences of inventory decisions.",
        "key_facts": [
            ("Scale", "Millions of SKUs"),
            ("Prime Day 2023", "Tens of millions of movements daily"),
            ("Challenge", "Stochastic lead times, correlated demand"),
            ("Trade-off", "Stockouts vs. excess inventory"),
            ("RL advantage", "Learns long-term consequences"),
            ("Unique complexity", "Demand correlations across products"),
        ],
        "questions": [
            {
                "question": "Why is inventory management naturally suited to reinforcement learning?",
                "notes": [
                    "Sequential decisions with delayed consequences",
                    "Today's order affects next week's availability",
                    "State space is complex (inventory levels, demand forecasts)",
                    "Reward is long-term (profit, not just today's sales)"
                ]
            },
            {
                "question": "Amazon balances stockouts (lost sales) against overstock (waste). How should these be weighted?",
                "notes": [
                    "Stockout cost: Lost sale + customer frustration + competitor switch",
                    "Overstock cost: Storage + depreciation + possible liquidation",
                    "Varies by product (perishables vs. electronics)",
                    "Customer lifetime value matters more than single transaction"
                ]
            },
            {
                "question": "Their system handles 'correlated demand'—when one product's sales affect another's. What examples of correlated demand can you think of?",
                "notes": [
                    "Complements: Phones and cases, printers and ink",
                    "Substitutes: Coke and Pepsi (negative correlation)",
                    "Bundles: Items often bought together",
                    "Seasonal: Many products spike together (back-to-school)"
                ]
            },
            {
                "question": "How might Amazon's approach differ from a traditional retailer with fewer SKUs?",
                "notes": [
                    "Traditional: Rules-based (reorder when stock < X)",
                    "Amazon: Learning-based (predict, adapt, optimize)",
                    "Scale makes ML investment worthwhile",
                    "Traditional retailers now adopting similar approaches"
                ]
            }
        ],
        "key_insight": "RL excels when there are sequential decisions with delayed consequences. Inventory management is a natural fit because today's decisions affect tomorrow's outcomes.",
        "sources": [
            {"name": "Amazon Science: Reinforcement Learning for Inventory", "url": "https://www.amazon.science/publications/deep-reinforcement-learning-for-inventory-management"},
            {"name": "AWS: Supply Chain Optimization", "url": "https://aws.amazon.com/solutions/retail/supply-chain-management/"},
            {"name": "Operations Research: Deep RL for Inventory Control", "url": "https://pubsonline.informs.org/doi/10.1287/opre.2021.2126"},
        ],
        "teaching_tips": [
            "Good case for connecting RL to business operations",
            "Ask: What would you reorder if you were Amazon?",
            "Connect to lecture: State = inventory, Action = order, Reward = profit",
            "Can discuss how COVID disrupted supply chains (non-stationary)"
        ]
    },
    # WEEK 5
    {
        "week": 5,
        "topic": "Neural Networks",
        "id": "Week5A_AlphaFold",
        "title": "AlphaFold: Solving Biology's 50-Year Grand Challenge",
        "summary": "For 50 years, scientists struggled to predict how proteins fold into 3D shapes from their amino acid sequences. In 2020, DeepMind's AlphaFold achieved accuracy comparable to experimental methods, effectively solving the problem. DeepMind then released 200+ million predicted protein structures for free—covering nearly every known protein. Drug companies now use AlphaFold to accelerate research that previously took months or years. The question: why did DeepMind give away such valuable IP?",
        "key_facts": [
            ("Problem duration", "50 years unsolved"),
            ("Breakthrough", "2020 (CASP14 competition)"),
            ("Structures released", "200+ million (free)"),
            ("Coverage", "Nearly every known protein"),
            ("Business model", "Open-sourced, not monetized directly"),
            ("Impact", "Drug discovery accelerated months to days"),
        ],
        "questions": [
            {
                "question": "AlphaFold solved a problem biologists worked on for 50 years. What made it tractable for neural networks?",
                "notes": [
                    "Large dataset: Protein Data Bank had 180,000+ structures",
                    "Clear objective: Predict atomic coordinates",
                    "Physical constraints: Chemistry rules narrow possibilities",
                    "Attention mechanism captures long-range dependencies"
                ]
            },
            {
                "question": "DeepMind made AlphaFold freely available. What are the business implications of open-sourcing such valuable IP?",
                "notes": [
                    "PR and prestige value (talent attraction)",
                    "Accelerates scientific progress (goodwill)",
                    "AlphaFold itself isn't a product—it's a capability demonstration",
                    "Google/Alphabet values long-term research over short-term revenue"
                ]
            },
            {
                "question": "AlphaFold predicts structures but doesn't explain why proteins fold as they do. Is this 'understanding'? Does it matter?",
                "notes": [
                    "Prediction vs. explanation debate",
                    "Practical utility doesn't require understanding",
                    "Scientists still want to know mechanisms",
                    "Similar debate in all of ML"
                ]
            },
            {
                "question": "Drug companies are using AlphaFold to accelerate research. How might this change pharmaceutical competitive dynamics?",
                "notes": [
                    "Levels playing field (all have access)",
                    "Speed advantage goes to those who integrate fastest",
                    "Shifts bottleneck to other parts of drug development",
                    "Still need wet lab validation"
                ]
            }
        ],
        "key_insight": "Neural networks excel with abundant data and clear objectives—even for problems humans couldn't solve. Open-sourcing can accelerate scientific progress while demonstrating capabilities.",
        "sources": [
            {"name": "Nature: AlphaFold Solves Protein Folding", "url": "https://www.nature.com/articles/s41586-021-03819-2"},
            {"name": "DeepMind: AlphaFold", "url": "https://deepmind.google/technologies/alphafold/"},
            {"name": "AlphaFold Protein Structure Database", "url": "https://alphafold.ebi.ac.uk/"},
            {"name": "Science: AI Revolutionizes Biology", "url": "https://www.science.org/doi/10.1126/science.abm0101"},
        ],
        "teaching_tips": [
            "Nobel Prize 2024 in Chemistry went to AlphaFold—very current",
            "Show visualizations of protein structures if possible",
            "Good for discussing the 'black box' vs. understanding debate",
            "Connect to lecture: Attention mechanism (foreshadows Week 7)"
        ]
    },
    {
        "week": 5,
        "topic": "Neural Networks",
        "id": "Week5B_DeepMind",
        "title": "DeepMind: 40% Reduction in Data Center Cooling Costs",
        "summary": "Google's data centers consume massive amounts of energy, with cooling being a major component. DeepMind applied neural networks to optimize cooling systems, achieving a 40% reduction in cooling energy—translating to 15% reduction in overall Power Usage Effectiveness (PUE). The system takes 120 sensor inputs and recommends actions every 5 minutes, with humans reviewing before implementation. The success led to partnerships with UK National Grid for broader energy optimization.",
        "key_facts": [
            ("Energy reduction", "40% cooling, 15% overall PUE"),
            ("Inputs", "120 sensors"),
            ("Update frequency", "Recommendations every 5 minutes"),
            ("Safety approach", "Human review before implementation"),
            ("Expansion", "UK National Grid partnership"),
            ("Challenge", "Many interacting variables"),
        ],
        "questions": [
            {
                "question": "Data center cooling seems like a simple engineering problem. Why did neural networks outperform traditional optimization?",
                "notes": [
                    "120 sensors = high-dimensional input space",
                    "Complex interactions between variables",
                    "Non-linear relationships",
                    "Traditional control theory struggles with this complexity"
                ]
            },
            {
                "question": "Google saved 40% on cooling costs. What other industrial processes might benefit from similar approaches?",
                "notes": [
                    "Manufacturing (process optimization)",
                    "HVAC in large buildings",
                    "Chemical plants (yield optimization)",
                    "Energy grid management",
                    "Key requirement: sensors + clear objective"
                ]
            },
            {
                "question": "The model was trained on historical data. How would it handle unprecedented situations (e.g., new hardware)?",
                "notes": [
                    "Historical data has limits",
                    "New hardware = distribution shift",
                    "Needs retraining or transfer learning",
                    "Human review catches anomalies"
                ]
            },
            {
                "question": "DeepMind partnered with UK National Grid. What's different about applying AI to critical infrastructure?",
                "notes": [
                    "Failure is not an option (blackouts)",
                    "Regulatory constraints",
                    "Public accountability",
                    "Longer validation periods required"
                ]
            }
        ],
        "key_insight": "Neural networks can optimize complex systems with many interacting variables. Human oversight remains important for safety-critical applications.",
        "sources": [
            {"name": "DeepMind: AI for Data Center Cooling", "url": "https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/"},
            {"name": "Google: Machine Learning for Data Centers", "url": "https://blog.google/outreach-initiatives/environment/deepmind-ai-reduces-energy-used-for/"},
            {"name": "DeepMind: AI and the Grid", "url": "https://deepmind.google/discover/blog/machine-learning-can-boost-the-value-of-wind-energy/"},
        ],
        "teaching_tips": [
            "Good case for industrial AI applications",
            "Emphasize the human-in-the-loop approach",
            "Connect to lecture: Neural networks as function approximators",
            "Can discuss energy/climate implications"
        ]
    },
    {
        "week": 5,
        "topic": "Neural Networks",
        "id": "Week5C_HM",
        "title": "H&M: The $4 Billion Inventory Problem",
        "summary": "In 2018, H&M accumulated $4.3 billion in unsold inventory—partly due to forecasting failures. Fast fashion requires predicting trends months in advance with long supply chain lead times. Competitors like Zara take a different approach: shorter lead times and rapid response to actual sales. H&M has since invested heavily in AI for demand prediction, but the fundamental challenge remains: can any model predict fashion trends accurately enough to avoid massive waste?",
        "key_facts": [
            ("Unsold inventory (2018)", "$4.3 billion"),
            ("Root cause", "Forecasting failures + long lead times"),
            ("Zara's approach", "Short lead times, rapid response"),
            ("H&M's response", "Heavy AI investment"),
            ("Fundamental challenge", "Fashion trends are hard to predict"),
            ("Environmental impact", "Massive waste from overproduction"),
        ],
        "questions": [
            {
                "question": "Fast fashion requires predicting trends months in advance. Why is this particularly challenging?",
                "notes": [
                    "Fashion is driven by culture, celebrities, social media",
                    "Trends can emerge and die quickly",
                    "Long lead times mean committing before signals are clear",
                    "Even humans (designers) are often wrong"
                ]
            },
            {
                "question": "H&M's $4B inventory problem was partly a forecasting failure. What other factors might have contributed?",
                "notes": [
                    "Supply chain rigidity (can't adjust once ordered)",
                    "Competitive pressure to have full stores",
                    "Incentives: Buyers rewarded for stocked shelves, not accuracy",
                    "Shift to e-commerce (different demand patterns)"
                ]
            },
            {
                "question": "How should H&M balance AI predictions with human designer intuition?",
                "notes": [
                    "AI for base demand, humans for trend bets",
                    "Use AI to identify which human predictions are reliable",
                    "Combine both in ensemble approaches",
                    "Humans good at qualitative, AI at quantitative"
                ]
            },
            {
                "question": "What's the environmental cost of forecasting errors in fashion? How should this factor into model evaluation?",
                "notes": [
                    "Fashion is one of most polluting industries",
                    "Overproduction leads to landfills or incineration",
                    "Sustainability metrics in addition to profit",
                    "Consumers increasingly care about sustainability"
                ]
            }
        ],
        "key_insight": "Even sophisticated companies can fail at prediction—the stakes are high in inventory-heavy businesses. Sometimes the answer isn't better prediction but shorter feedback loops.",
        "sources": [
            {"name": "Business of Fashion: H&M's Inventory Crisis", "url": "https://www.businessoffashion.com/articles/retail/hm-has-a-4-billion-inventory-crisis/"},
            {"name": "Harvard Business Review: Zara's Fast Fashion Edge", "url": "https://hbr.org/2004/11/rapid-fire-fulfillment"},
            {"name": "MIT Sloan: H&M Uses AI for Demand Forecasting", "url": "https://mitsloan.mit.edu/ideas-made-to-matter/how-h-m-using-ai-and-algorithms"},
            {"name": "Reuters: H&M AI Investment", "url": "https://www.reuters.com/article/us-h-m-strategy-idUSKCN1G80DF"},
        ],
        "teaching_tips": [
            "Contrast H&M (better prediction) vs. Zara (better process)",
            "Good for discussing when prediction isn't the answer",
            "Connect to Week 1: Zillow also faced regime change",
            "Environmental angle can generate good discussion"
        ]
    },
    # WEEK 6
    {
        "week": 6,
        "topic": "Convolutional Neural Networks",
        "id": "Week6A_Tesla",
        "title": "Tesla Autopilot: The Vision-Only Bet",
        "summary": "Tesla is the only major autonomous vehicle company betting entirely on cameras—no LIDAR. Their HydraNet architecture fuses 8 cameras into a single neural network running 50+ perception tasks simultaneously. Training data comes from 1.5 petabytes of video (1 million clips, 6 billion labeled objects) collected from their customer fleet. The controversy: Tesla's 'Full Self-Driving' name suggests more capability than the system delivers, and crashes make headlines. But Tesla argues cameras are sufficient because humans drive with vision alone.",
        "key_facts": [
            ("Sensor approach", "8 cameras only (no LIDAR)"),
            ("Architecture", "HydraNet (50+ tasks)"),
            ("Training data", "1.5 petabytes, 6B labeled objects"),
            ("Data source", "Customer fleet (millions of cars)"),
            ("Controversy", "'Full Self-Driving' naming"),
            ("Tesla's argument", "Humans drive with vision alone"),
        ],
        "questions": [
            {
                "question": "Tesla bet on cameras while competitors use LIDAR. What are the trade-offs of each approach?",
                "notes": [
                    "Cameras: Cheap, scalable, but struggle in darkness/weather",
                    "LIDAR: Precise depth, works at night, but expensive ($10K+)",
                    "Tesla's bet: Neural networks can extract depth from video",
                    "Competitors' bet: Redundant sensors are safer"
                ]
            },
            {
                "question": "Tesla trains on data from its customer fleet. What advantages and risks does this create?",
                "notes": [
                    "Advantage: Millions of cars = massive data",
                    "Advantage: Real-world edge cases (not just test tracks)",
                    "Risk: Privacy concerns (always recording)",
                    "Risk: Selection bias (where Tesla owners drive)"
                ]
            },
            {
                "question": "Autopilot crashes make headlines. How should we evaluate safety—compared to human drivers or an absolute standard?",
                "notes": [
                    "Tesla claims fewer crashes per mile than average driver",
                    "But: Tesla drivers may be more attentive (know it's limited)",
                    "Absolute standard is impossible (no system is 100% safe)",
                    "Public expects higher standard from AI than humans"
                ]
            },
            {
                "question": "Tesla's 'Full Self-Driving' name is controversial. How should companies communicate AI capabilities and limitations?",
                "notes": [
                    "FSD is SAE Level 2 (driver must be attentive)",
                    "Name suggests Level 4 or 5 (fully autonomous)",
                    "Regulatory scrutiny increasing",
                    "Marketing vs. safety communication tension"
                ]
            }
        ],
        "key_insight": "Computer vision enables machines to perceive the world, but perception is not understanding. Different companies make fundamentally different bets on the path to autonomy.",
        "sources": [
            {"name": "Tesla AI Day Presentation", "url": "https://www.youtube.com/watch?v=j0z4FweCy4M"},
            {"name": "IEEE Spectrum: Tesla's Computer Vision Approach", "url": "https://spectrum.ieee.org/tesla-autopilot-computer-vision"},
            {"name": "Ars Technica: Tesla FSD Review", "url": "https://arstechnica.com/cars/2023/08/teslas-full-self-driving-mode-is-still-a-dangerous-lie/"},
            {"name": "NHTSA: Tesla Autopilot Investigation", "url": "https://www.nhtsa.gov/press-releases/nhtsa-closes-investigation-tesla-autopilot"},
        ],
        "teaching_tips": [
            "Very current and polarizing—students will have opinions",
            "Good for discussing sensor fusion vs. single-modality approaches",
            "Connect to lecture: CNNs processing camera images",
            "Can discuss the marketing vs. capability gap"
        ]
    },
    {
        "week": 6,
        "topic": "Convolutional Neural Networks",
        "id": "Week6B_Waymo",
        "title": "Waymo: The LIDAR Approach",
        "summary": "Waymo takes the opposite bet from Tesla: sensors everywhere. Their vehicles use LIDAR, cameras, and radar for redundant perception. EMMA (End-to-End Multimodal Model) uses Google's Gemini LLM to generate driving trajectories directly from sensor data. Waymo explored 10,000+ CNN architectures to find optimal networks. Their robotaxis have driven 20+ million autonomous miles with zero at-fault fatalities. The trade-off: expensive sensors limit scaling, but safety record is unmatched.",
        "key_facts": [
            ("Sensor approach", "LIDAR + cameras + radar"),
            ("EMMA", "Uses Gemini LLM for trajectories"),
            ("Architecture search", "10,000+ CNN variants tested"),
            ("Autonomous miles", "20+ million"),
            ("Safety record", "Zero at-fault fatalities"),
            ("Trade-off", "Expensive sensors limit scaling"),
        ],
        "questions": [
            {
                "question": "Waymo uses LIDAR while Tesla doesn't. What does each company believe about the path to autonomous driving?",
                "notes": [
                    "Waymo: Redundancy and safety first, cost will decrease",
                    "Tesla: Scale data collection, software will improve",
                    "Waymo: Start in geofenced areas, expand carefully",
                    "Tesla: Deploy everywhere, learn from edge cases"
                ]
            },
            {
                "question": "Waymo operates robotaxis in limited areas. Tesla aims for everywhere. Which strategy is better?",
                "notes": [
                    "Waymo: Prove safety before scaling",
                    "Tesla: Scale first, improve with data",
                    "Waymo: Revenue-generating service now",
                    "Tesla: Consumer product, broader market"
                ]
            },
            {
                "question": "EMMA uses an LLM for driving decisions. What are the implications of language models controlling vehicles?",
                "notes": [
                    "LLMs can reason about complex scenarios",
                    "But: Hallucination risk in safety-critical settings",
                    "Multimodal input: Sensors + language understanding",
                    "Represents convergence of CV and NLP"
                ]
            },
            {
                "question": "Waymo has driven 20+ million autonomous miles. How much data is 'enough' to prove safety?",
                "notes": [
                    "Statistically: Need billions of miles for rare events",
                    "Human baseline: ~1 fatality per 100M miles",
                    "20M miles is good, but edge cases are rare",
                    "Simulation helps but isn't the same"
                ]
            }
        ],
        "key_insight": "Different companies make fundamentally different bets on the path to autonomy. Waymo prioritizes safety and redundancy; Tesla prioritizes scale and data.",
        "sources": [
            {"name": "Waymo: Safety Report", "url": "https://waymo.com/safety/"},
            {"name": "Waymo Blog: EMMA Announcement", "url": "https://waymo.com/blog/"},
            {"name": "Ars Technica: Waymo's Approach to Autonomy", "url": "https://arstechnica.com/cars/2023/03/after-years-of-testing-waymo-is-ready-for-its-close-up/"},
            {"name": "The Verge: Waymo Robotaxi Review", "url": "https://www.theverge.com/2023/8/2/23816934/waymo-driverless-robotaxi-ride-phoenix"},
        ],
        "teaching_tips": [
            "Pair with Tesla case for great compare/contrast",
            "Good for discussing different AI strategies",
            "Connect to lecture: Sensor fusion, multi-task learning",
            "Ask: Which company's approach would you invest in?"
        ]
    },
    {
        "week": 6,
        "topic": "Convolutional Neural Networks",
        "id": "Week6C_JohnDeere",
        "title": "John Deere See & Spray: Precision Agriculture",
        "summary": "John Deere paid $305 million to acquire Blue River Technology and its See & Spray system. Using computer vision, the system identifies individual weeds and sprays only them—reducing herbicide use by 77%. The technology processes 3,000 spray decisions per minute at highway speeds. For farmers, this means lower chemical costs and environmental impact. For John Deere, it means data on millions of acres—a strategic asset that competitors can't easily replicate.",
        "key_facts": [
            ("Acquisition price", "$305 million (Blue River)"),
            ("Herbicide reduction", "77%"),
            ("Processing speed", "3,000 decisions per minute"),
            ("Operating speed", "Highway speeds"),
            ("Strategic asset", "Data on millions of acres"),
            ("Farmer benefit", "Lower costs, less environmental impact"),
        ],
        "questions": [
            {
                "question": "John Deere paid $305M for Blue River. What were they buying—technology, talent, or data?",
                "notes": [
                    "Technology: See & Spray system",
                    "Talent: Computer vision expertise",
                    "Data: Training data for agricultural CV",
                    "Position: Leadership in precision ag",
                    "All four, but data moat may be most valuable long-term"
                ]
            },
            {
                "question": "See & Spray reduces herbicide 77%. Who captures the value—farmers, John Deere, or society?",
                "notes": [
                    "Farmers: Lower input costs",
                    "John Deere: Premium pricing on equipment",
                    "Society: Environmental benefits",
                    "Value distribution depends on market structure"
                ]
            },
            {
                "question": "Farmers have been skeptical of 'precision ag' promises. What's different about computer vision?",
                "notes": [
                    "Previous tech: GPS guidance, variable rate (incremental)",
                    "Computer vision: Visible, immediate impact",
                    "77% reduction is dramatic (not marginal improvement)",
                    "Can see the sprayer working differently"
                ]
            },
            {
                "question": "John Deere now has data on millions of acres. What are the strategic implications of this data moat?",
                "notes": [
                    "Improve models faster than competitors",
                    "Cross-sell other services (yield prediction, etc.)",
                    "Lock-in effect (leaving means losing data)",
                    "Potential for data marketplace"
                ]
            }
        ],
        "key_insight": "Computer vision transforms industries by making visual inspection scalable. The equipment sale is the beginning of a data relationship.",
        "sources": [
            {"name": "John Deere: See & Spray", "url": "https://www.deere.com/en/sprayers/see-spray/"},
            {"name": "Wired: John Deere's AI Farming Revolution", "url": "https://www.wired.com/story/john-deere-see-spray-ai-farming/"},
            {"name": "MIT Technology Review: Precision Agriculture", "url": "https://www.technologyreview.com/2023/05/09/1072770/john-deere-ai-farming/"},
            {"name": "Blue River Technology (archived)", "url": "https://www.bluerivertechnology.com/"},
        ],
        "teaching_tips": [
            "Good case for non-tech industry AI application",
            "Data moat discussion connects to strategy",
            "Connect to lecture: Real-time CNN inference",
            "Environmental angle can generate discussion"
        ]
    },
    {
        "week": 6,
        "topic": "Convolutional Neural Networks",
        "id": "Week6D_AmazonGo",
        "title": "Amazon Go: Cashierless Checkout",
        "summary": "Amazon Go's 'Just Walk Out' technology uses computer vision, sensor fusion, and deep learning to enable cashierless checkout. Customers scan their app, take items, and leave—no checkout lines. 84% of surveyed customers prefer it to traditional shopping. The catch: the technology requires approximately $1 million in hardware investment per store. Amazon has since pivoted, licensing 'Just Walk Out' to other retailers rather than scaling their own stores.",
        "key_facts": [
            ("Technology", "Computer vision + sensor fusion + deep learning"),
            ("Customer preference", "84% prefer to traditional"),
            ("Hardware cost", "~$1M per store"),
            ("Pivot", "Licensing to other retailers"),
            ("Privacy concern", "Continuous customer tracking"),
            ("Jobs impact", "Eliminates cashiers, creates technicians"),
        ],
        "questions": [
            {
                "question": "Amazon Go requires $1M+ per store in hardware. When does this investment make sense?",
                "notes": [
                    "High-traffic locations (airports, stadiums)",
                    "High labor cost markets",
                    "Premium brand positioning",
                    "Data collection value may justify cost"
                ]
            },
            {
                "question": "The technology tracks customers throughout the store. What privacy concerns does this raise?",
                "notes": [
                    "Continuous surveillance while shopping",
                    "Biometric data collection (face, body)",
                    "Linking physical behavior to digital identity",
                    "Data retention and security questions"
                ]
            },
            {
                "question": "Amazon licensed 'Just Walk Out' to other retailers. Why sell the technology rather than keep it proprietary?",
                "notes": [
                    "Amazon Go stores weren't scaling (too expensive)",
                    "Better to profit from technology than from stores",
                    "Platform play: Be the infrastructure",
                    "More data from more stores improves models"
                ]
            },
            {
                "question": "What jobs are eliminated or created by cashierless stores?",
                "notes": [
                    "Eliminated: Cashiers, some stock clerks",
                    "Created: Technicians, data analysts, customer service",
                    "Net effect: Likely fewer jobs",
                    "Remaining jobs require different skills"
                ]
            }
        ],
        "key_insight": "Computer vision enables new business models, but infrastructure costs matter. Technology advantages often become platform/licensing businesses.",
        "sources": [
            {"name": "Amazon: Just Walk Out Technology", "url": "https://justwalkout.com/"},
            {"name": "TechCrunch: Amazon Go's Pivot", "url": "https://techcrunch.com/2023/04/04/amazon-reportedly-to-stop-opening-new-amazon-go-stores/"},
            {"name": "CNBC: Inside Amazon Go", "url": "https://www.cnbc.com/2018/01/22/amazon-go-cashier-less-store-how-it-works.html"},
            {"name": "Vox: Amazon Go and Retail Jobs", "url": "https://www.vox.com/recode/2020/2/25/21147068/amazon-go-grocery-store-expansion-jobs"},
        ],
        "teaching_tips": [
            "Good for discussing technology commercialization strategies",
            "Privacy discussion is very relevant",
            "Connect to lecture: Multi-camera CV systems",
            "Ask: Would you shop at a store that tracks your every move?"
        ]
    },
    # WEEK 7
    {
        "week": 7,
        "topic": "Transformers & LLMs",
        "id": "Week7A_Klarna",
        "title": "Klarna's AI Reversal: When Automation Goes Too Far",
        "summary": "In February 2024, Klarna announced their AI assistant handled 2.3 million conversations in its first month—'equivalent to the work of 700 full-time agents.' Resolution time dropped from 11 minutes to 2 minutes. Media celebrated the efficiency gains. Then came the quiet reversal. By late 2024, CEO Sebastian Siemiatkowski admitted: 'We focused too much on efficiency. The result was lower quality.' Klarna rehired human agents and repositioned AI as support for humans, not replacement.",
        "key_facts": [
            ("Initial claim", "AI = 700 agents"),
            ("Conversations handled", "2.3 million (first month)"),
            ("Resolution time", "11 min → 2 min"),
            ("The reversal", "Admitted quality suffered"),
            ("New approach", "AI supports humans, not replaces"),
            ("Lesson learned", "Efficiency ≠ quality"),
        ],
        "questions": [
            {
                "question": "Klarna initially reported their AI did the work of 700 agents. What metrics might have hidden quality problems?",
                "notes": [
                    "Resolution time doesn't measure resolution quality",
                    "Conversation count doesn't measure satisfaction",
                    "Cost savings metric ignores customer experience",
                    "Short-term metrics can miss long-term damage"
                ]
            },
            {
                "question": "Later, Klarna admitted quality suffered. What might have gone wrong that wasn't captured in resolution time?",
                "notes": [
                    "Customers got fast but wrong answers",
                    "Complex issues weren't actually resolved",
                    "Customer frustration led to churn",
                    "Repeat contacts (same issue, multiple conversations)"
                ]
            },
            {
                "question": "Klarna now uses AI for routine queries with humans always available. How would you design the handoff?",
                "notes": [
                    "AI handles simple queries (balance, payment dates)",
                    "Escalation triggers: Negative sentiment, repeated questions",
                    "Human review for high-stakes decisions",
                    "Customer can always request human"
                ]
            },
            {
                "question": "What metrics beyond cost savings should companies track for AI customer service?",
                "notes": [
                    "Customer satisfaction (CSAT)",
                    "Net Promoter Score (NPS)",
                    "First contact resolution rate",
                    "Repeat contact rate",
                    "Customer lifetime value impact"
                ]
            }
        ],
        "key_insight": "The gap between 'demo' and 'production' is real. Customer-facing AI requires careful monitoring of quality, not just efficiency metrics.",
        "sources": [
            {"name": "Klarna: AI Assistant Announcement", "url": "https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/"},
            {"name": "Business Insider: Klarna's AI Reversal", "url": "https://www.businessinsider.com/klarna-ceo-ai-customer-service-quality-2024-9"},
            {"name": "Customer Experience Dive: Lessons from Klarna", "url": "https://www.customerexperiencedive.com/"},
            {"name": "Forbes: AI in Customer Service Pitfalls", "url": "https://www.forbes.com/sites/forbestechcouncil/2024/03/15/what-klarnas-ai-pivot-teaches-us-about-customer-service/"},
        ],
        "teaching_tips": [
            "Very current case (2024)—students may have seen headlines",
            "Good for discussing metric selection and Goodhart's Law",
            "Connect to lecture: LLM capabilities and limitations",
            "Ask: Have you had a frustrating chatbot experience?"
        ]
    },
    {
        "week": 7,
        "topic": "Transformers & LLMs",
        "id": "Week7B_JPMorgan_COIN",
        "title": "JPMorgan COIN: Contract Intelligence",
        "summary": "JPMorgan's COIN (Contract Intelligence) system uses NLP to review commercial loan agreements. Before COIN, lawyers spent 360,000 hours annually on this task. Now it takes seconds. The system extracts terms, interprets obligations, identifies risks, and detects ambiguity. But contracts have real legal consequences. When the AI misses a critical term or misinterprets an obligation, who bears the liability? And what happens to the lawyers whose work was automated?",
        "key_facts": [
            ("Previous time", "360,000 hours annually"),
            ("Current time", "Seconds"),
            ("Capabilities", "Extract terms, detect ambiguity, assess risk"),
            ("Document type", "Commercial loan agreements"),
            ("Liability question", "Who's responsible for AI errors?"),
            ("Workforce impact", "Lawyer roles transformed"),
        ],
        "questions": [
            {
                "question": "COIN saved 360,000 hours of lawyer time. What happened to those lawyers and their roles?",
                "notes": [
                    "Shifted to higher-value work (negotiation, strategy)",
                    "More deals processed with same headcount",
                    "Junior lawyer roles most affected",
                    "New roles: AI training, validation, exception handling"
                ]
            },
            {
                "question": "Contract review requires understanding legal nuance. How confident should we be in AI interpretations?",
                "notes": [
                    "Standard terms: Very high confidence",
                    "Unusual clauses: Needs human review",
                    "Ambiguity detection is valuable (flags for human)",
                    "AI as first pass, human for validation"
                ]
            },
            {
                "question": "What's the liability when AI misses a critical contract term? Who bears responsibility?",
                "notes": [
                    "Currently: Lawyers still sign off (retain liability)",
                    "Future: Unclear as AI role expands",
                    "Insurance implications",
                    "Regulatory guidance evolving"
                ]
            },
            {
                "question": "Which other professional services might be transformed by NLP? Which are resistant?",
                "notes": [
                    "Transformable: Due diligence, compliance, auditing",
                    "Partially: Medical records, patent search",
                    "Resistant: Courtroom advocacy, client relationships",
                    "Key: Document-heavy, pattern-recognizable tasks"
                ]
            }
        ],
        "key_insight": "NLP can automate document-heavy professional work, but stakes are high when mistakes have legal consequences. Human oversight remains essential.",
        "sources": [
            {"name": "JPMorgan: COIN Announcement", "url": "https://www.jpmorgan.com/technology/technology-blog/machine-learning-and-nlp"},
            {"name": "Bloomberg: JPMorgan's Robot Lawyers", "url": "https://www.bloomberg.com/news/articles/2017-02-28/jpmorgan-marshals-an-army-of-developers-to-automate-high-finance"},
            {"name": "Harvard Law: AI in Legal Practice", "url": "https://hls.harvard.edu/today/a-new-era-of-ai-in-law/"},
            {"name": "Reuters: AI Legal Tools", "url": "https://www.reuters.com/legal/transactional/ai-legal-tech-tools-are-getting-better-so-are-risks-2023-04-03/"},
        ],
        "teaching_tips": [
            "Good for discussing professional services automation",
            "Connects to broader workforce transformation discussion",
            "Connect to lecture: NLP for document understanding",
            "Ask: What would you want AI to review in a contract you're signing?"
        ]
    },
    {
        "week": 7,
        "topic": "Transformers & LLMs",
        "id": "Week7C_Alexa",
        "title": "Amazon Alexa: Voice AI at Scale",
        "summary": "Alexa processes millions of voice requests daily through a complex AI pipeline: 6-microphone array for noise cancellation, speech-to-text conversion, natural language understanding for intent inference, and text-to-speech via Polly. Research has improved prediction accuracy 9-30% through automatic domain discovery and personalization. But despite $10+ billion in investment, Alexa reportedly loses money on every device. Voice assistants haven't achieved the adoption some predicted. What's holding them back?",
        "key_facts": [
            ("Daily requests", "Millions"),
            ("Microphone array", "6 mics (far-field recognition)"),
            ("Pipeline stages", "ASR → NLU → Dialog → TTS"),
            ("Accuracy improvement", "9-30% via research"),
            ("Investment", "$10+ billion"),
            ("Business model", "Reportedly loses money per device"),
        ],
        "questions": [
            {
                "question": "Alexa processes millions of requests daily. What makes voice interfaces harder than text?",
                "notes": [
                    "Ambient noise and multiple speakers",
                    "Accents and speech variations",
                    "No visual context (can't point)",
                    "Real-time processing requirements",
                    "Wake word detection while sleeping"
                ]
            },
            {
                "question": "Amazon collects massive amounts of voice data. What are the privacy implications?",
                "notes": [
                    "Always listening for wake word",
                    "Recordings stored for improvement",
                    "Human reviewers listen to samples",
                    "Potential for misuse or breaches"
                ]
            },
            {
                "question": "Alexa works in many languages and accents. How would you prioritize which to support?",
                "notes": [
                    "Market size (speakers × purchasing power)",
                    "Existing Echo device distribution",
                    "Data availability for training",
                    "Strategic markets for Amazon"
                ]
            },
            {
                "question": "Voice assistants haven't achieved the adoption some predicted. What's holding them back?",
                "notes": [
                    "Limited utility beyond timers and music",
                    "Awkward for complex queries",
                    "Privacy concerns",
                    "Screens often more efficient",
                    "Voice isn't natural for many tasks"
                ]
            }
        ],
        "key_insight": "Voice AI requires solving multiple hard problems simultaneously—and commercial success requires more than technical excellence. Great technology doesn't guarantee product-market fit.",
        "sources": [
            {"name": "Amazon Science: Alexa Research", "url": "https://www.amazon.science/tag/alexa"},
            {"name": "Ars Technica: How Alexa Works", "url": "https://arstechnica.com/gadgets/2019/08/how-amazons-alexa-went-from-a-smart-speaker-to-the-center-of-your-home/"},
            {"name": "Business Insider: Alexa Losses", "url": "https://www.businessinsider.com/amazon-alexa-10-billion-loss-2022-11"},
            {"name": "WSJ: The Voice Assistant Wars", "url": "https://www.wsj.com/articles/amazon-alexa-google-assistant-apple-siri-11674762042"},
        ],
        "teaching_tips": [
            "Many students will have experience with Alexa/Siri/Google",
            "Good for discussing the gap between tech and product",
            "Connect to lecture: ASR, NLU, dialog systems",
            "Can discuss why LLMs might change voice assistants"
        ]
    },
    # WEEK 8
    {
        "week": 8,
        "topic": "Generative AI & Agents",
        "id": "Week8A_MorganStanley",
        "title": "Morgan Stanley: Enterprise AI at Scale",
        "summary": "Morgan Stanley built an AI assistant using GPT-4 fine-tuned on 100,000+ internal research documents. 16,000 financial advisors can now query firm-specific content instantly—market analysis, investment strategies, client talking points. One managing director said: 'The barrier between knowledge and communication has disappeared.' But this raises questions: What happens when AI confidently delivers outdated information? And if junior analysts previously synthesized research, what's their role now?",
        "key_facts": [
            ("Model", "GPT-4 (fine-tuned)"),
            ("Training documents", "100,000+ internal"),
            ("Users", "16,000 financial advisors"),
            ("Use cases", "Research queries, client prep, analysis"),
            ("Key quote", "'Barrier between knowledge and communication has disappeared'"),
            ("Challenge", "Outdated or conflicting information"),
        ],
        "questions": [
            {
                "question": "Morgan Stanley's AI answers based on internal research. What are the risks of outdated or conflicting information?",
                "notes": [
                    "Markets change faster than documents update",
                    "Different analysts may have conflicting views",
                    "AI may present outdated view as current",
                    "Need version control and recency awareness"
                ]
            },
            {
                "question": "Junior analysts previously synthesized research. How does AI change their role?",
                "notes": [
                    "Less 'summarizing' work",
                    "More 'validating AI output' work",
                    "Shift to higher-value analysis",
                    "Training path changes (how do you learn?)"
                ]
            },
            {
                "question": "Morgan Stanley chose to fine-tune rather than use off-the-shelf AI. What factors drive build-vs-buy?",
                "notes": [
                    "Proprietary data advantage",
                    "Regulatory requirements (data can't leave firm)",
                    "Customization needs",
                    "Cost vs. capability trade-off"
                ]
            },
            {
                "question": "If you were a competing firm, how would you respond to this AI advantage?",
                "notes": [
                    "Build similar capability (table stakes)",
                    "Differentiate on quality of underlying research",
                    "Focus on areas AI can't help (relationships, judgment)",
                    "Find AI applications competitors haven't explored"
                ]
            }
        ],
        "key_insight": "Enterprise AI requires data strategy, integration, and governance—technology is the easy part. Competitive advantage comes from proprietary data and integration depth.",
        "sources": [
            {"name": "Morgan Stanley: AI Assistant Announcement", "url": "https://www.morganstanley.com/articles/morgan-stanley-openai-collaboration"},
            {"name": "CNBC: Morgan Stanley's AI for Advisors", "url": "https://www.cnbc.com/2023/09/18/morgan-stanley-chatgpt-financial-advisors.html"},
            {"name": "Harvard Business Review: AI in Financial Services", "url": "https://hbr.org/2023/07/how-generative-ai-will-change-the-jobs-of-financial-advisors"},
            {"name": "Financial Times: Banks Embrace AI", "url": "https://www.ft.com/content/morgan-stanley-ai-wealth-management"},
        ],
        "teaching_tips": [
            "Good capstone case—brings together many course themes",
            "Discuss enterprise AI challenges vs. consumer AI",
            "Connect to lecture: RAG, fine-tuning, enterprise deployment",
            "Ask: What would you want an AI assistant to help with?"
        ]
    },
    {
        "week": 8,
        "topic": "Generative AI & Agents",
        "id": "Week8B_Copilot",
        "title": "GitHub Copilot: AI Pair Programming",
        "summary": "Microsoft ran randomized controlled trials with 4,000+ developers to measure Copilot's impact. Results: 26% increase in overall productivity, 55.8% faster task completion, and 75% higher job satisfaction. Notably, less experienced developers benefited most. But Copilot was trained on public GitHub code, raising IP questions. And if AI can write code, what skills become more or less valuable for developers?",
        "key_facts": [
            ("Study size", "4,000+ developers (RCT)"),
            ("Productivity increase", "26% overall"),
            ("Task completion", "55.8% faster"),
            ("Job satisfaction", "75% higher"),
            ("Key finding", "Junior developers benefit most"),
            ("Training data", "Public GitHub repositories"),
        ],
        "questions": [
            {
                "question": "Less experienced developers benefited most from Copilot. What does this imply for training and hiring?",
                "notes": [
                    "AI as learning accelerator",
                    "Less emphasis on syntax memorization",
                    "More emphasis on problem decomposition",
                    "Hiring: Look for reasoning, not rote coding"
                ]
            },
            {
                "question": "Copilot was trained on public code. What are the IP implications of AI-generated code?",
                "notes": [
                    "Does output infringe on training data licenses?",
                    "Who owns AI-generated code?",
                    "Active lawsuits pending",
                    "Companies developing policies"
                ]
            },
            {
                "question": "Developers report 75% higher job satisfaction with Copilot. Why might this be?",
                "notes": [
                    "Less tedious boilerplate",
                    "More time on interesting problems",
                    "Reduced context switching",
                    "Feels like having a helpful partner"
                ]
            },
            {
                "question": "If AI can write code, what skills become more or less valuable for developers?",
                "notes": [
                    "Less valuable: Syntax memorization, boilerplate writing",
                    "More valuable: Architecture, problem decomposition",
                    "More valuable: Reviewing and validating AI output",
                    "More valuable: Understanding what to build"
                ]
            }
        ],
        "key_insight": "AI tools change the skill mix required—less syntax, more architecture and problem-solving. Rigorous evaluation (RCT) provides credible evidence of impact.",
        "sources": [
            {"name": "Microsoft Research: Copilot Productivity Study", "url": "https://arxiv.org/abs/2302.06590"},
            {"name": "GitHub: Copilot", "url": "https://github.com/features/copilot"},
            {"name": "ACM: AI Pair Programmers", "url": "https://cacm.acm.org/magazines/2023/7/274065-how-effective-are-ai-code-assistants/fulltext"},
            {"name": "The Verge: Copilot Lawsuit", "url": "https://www.theverge.com/2022/11/8/23446821/microsoft-openai-github-copilot-class-action-lawsuit-ai-copyright-violation-training-data"},
        ],
        "teaching_tips": [
            "Many CS students will have used Copilot",
            "Good for discussing AI's impact on knowledge work",
            "Connect to lecture: Code as language, LLMs for code",
            "RCT methodology is notable—discuss what makes evidence credible"
        ]
    },
    {
        "week": 8,
        "topic": "Generative AI & Agents",
        "id": "Week8C_Nike",
        "title": "Nike: AI-Driven Retail Transformation",
        "summary": "Since 2018, Nike acquired four AI companies: Celect (demand prediction), Zodiac (customer lifetime value), Datalogue (data integration), and Invertex (3D foot scanning). Total investment: $1+ billion. Nike Digital grew from 10% to 26% of revenue (2019-2023). But in 2024, Nike announced 1,600 layoffs and acknowledged over-rotating to digital at the expense of wholesale relationships. The AI transformation delivered metrics but created new problems.",
        "key_facts": [
            ("AI acquisitions", "4 companies since 2018"),
            ("Investment", "$1+ billion"),
            ("Digital revenue", "10% → 26% (2019-2023)"),
            ("2024 layoffs", "1,600 employees"),
            ("Problem acknowledged", "Over-rotated to digital"),
            ("Acquisitions", "Celect, Zodiac, Datalogue, Invertex"),
        ],
        "questions": [
            {
                "question": "Nike acquired 4 AI companies. When does it make sense to buy vs. build AI capabilities?",
                "notes": [
                    "Buy: Speed to market, acquire talent, proven technology",
                    "Build: Unique needs, core competency, control",
                    "Nike bought because retail AI wasn't core competency",
                    "Integration is the hard part"
                ]
            },
            {
                "question": "Zodiac predicts individual customer lifetime value. How might this change marketing and service?",
                "notes": [
                    "Allocate marketing spend to high-value prospects",
                    "Personalize offers based on predicted value",
                    "Service prioritization (VIPs get better support)",
                    "Risk: Self-fulfilling prophecy"
                ]
            },
            {
                "question": "Nike Digital grew from 10% to 26% of sales, but they acknowledged over-rotating. What went wrong?",
                "notes": [
                    "Neglected wholesale relationships (Foot Locker, etc.)",
                    "D2C requires different inventory/logistics",
                    "Lost shelf space and brand presence",
                    "Digital customers may be different (cherry-picking deals)"
                ]
            },
            {
                "question": "What AI capabilities would Nike's competitors need to match this strategy?",
                "notes": [
                    "Demand forecasting (reduce inventory risk)",
                    "Personalization (customer engagement)",
                    "Supply chain optimization",
                    "Competitors: Adidas, Under Armour also investing"
                ]
            }
        ],
        "key_insight": "AI strategy isn't just about technology—it's about acquiring capabilities and integrating them into the business. Success metrics can mask strategic problems.",
        "sources": [
            {"name": "Nike: Digital Transformation Strategy", "url": "https://investors.nike.com/"},
            {"name": "Retail Dive: Nike's AI Acquisitions", "url": "https://www.retaildive.com/news/nike-ai-machine-learning-acquisitions-celect-zodiac-datalogue-invertex/"},
            {"name": "Fortune: Nike's Digital Pivot Problems", "url": "https://fortune.com/2024/02/15/nike-layoffs-digital-transformation/"},
            {"name": "Harvard Business Review: Nike's Direct-to-Consumer Strategy", "url": "https://hbr.org/2020/12/what-nike-got-right-and-wrong-about-d2c"},
        ],
        "teaching_tips": [
            "Good closing case—shows both success and failure",
            "Connects AI to broader business strategy",
            "Nike brand is familiar to all students",
            "Ask: What would you have done differently?"
        ]
    },
]

# ============================================
# GENERATE ALL HANDOUTS
# ============================================

import os

# Create directories if needed
os.makedirs("/sessions/bold-upbeat-keller/mnt/CLAUDE/case-handouts/student", exist_ok=True)
os.makedirs("/sessions/bold-upbeat-keller/mnt/CLAUDE/case-handouts/instructor", exist_ok=True)

for case in cases:
    # Student handout
    student_file = f"/sessions/bold-upbeat-keller/mnt/CLAUDE/case-handouts/student/{case['id']}.pdf"
    create_student_handout(
        student_file,
        case['week'],
        case['topic'],
        case['title'],
        case['summary'],
        case['key_facts'],
        case['questions'],
        case['sources']
    )

    # Instructor notes
    instructor_file = f"/sessions/bold-upbeat-keller/mnt/CLAUDE/case-handouts/instructor/{case['id']}_INSTRUCTOR.pdf"
    create_instructor_notes(
        instructor_file,
        case['week'],
        case['topic'],
        case['title'],
        case['questions'],
        case['key_insight'],
        case.get('teaching_tips', [])
    )

print("\n✅ All handouts generated successfully!")
print("📁 Student handouts: /sessions/bold-upbeat-keller/mnt/CLAUDE/case-handouts/student/")
print("📁 Instructor notes: /sessions/bold-upbeat-keller/mnt/CLAUDE/case-handouts/instructor/")
