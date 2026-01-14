import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const TOTAL_SEATS = 75;
const TOTAL_DAYS = 180;
const ARRIVAL_RATE = 1; // Poisson lambda
const PRICES = [199, 249, 299, 349, 499, 699];

// Sample from Poisson distribution
const samplePoisson = (lambda) => {
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= Math.random();
  } while (p > L);
  return k - 1;
};

const getCapacityBucket = (remaining, total) => {
  const pct = remaining / total;
  if (pct > 0.7) return 'high';
  if (pct > 0.3) return 'medium';
  return 'low';
};

const getTimeBucket = (daysLeft) => {
  if (daysLeft >= 60) return 'far';
  if (daysLeft >= 15) return 'medium';
  return 'near';
};

const getState = (seatsRemaining, daysLeft) => getCapacityBucket(seatsRemaining, TOTAL_SEATS) + '-' + getTimeBucket(daysLeft);

const generateCustomerWTP = (daysLeft) => {
  const t = TOTAL_DAYS - daysLeft + 1;
  const meanWTP = 100 + t + Math.pow(t / 50, 5);
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return Math.max(0, meanWTP + z * 10);
};

const initQTable = () => {
  const Q = {};
  ['high', 'medium', 'low'].forEach(cap => {
    ['far', 'medium', 'near'].forEach(time => {
      // Optimistic initialization: start all Q-values high to encourage exploration
      PRICES.forEach(price => { Q[cap + '-' + time + '-' + price] = 20000; });
    });
  });
  return Q;
};

const getQValue = (Q, state, price) => Q[state + '-' + price] || 0;
const setQValue = (Q, state, price, value) => { Q[state + '-' + price] = value; };

const initBanditState = () => {
  const state = { totalFlights: 0 };
  // For Thompson Sampling per flight: track total revenue and number of flights per price
  // Model revenue with Gaussian posterior (approximate)
  PRICES.forEach(price => { state[price] = { totalRevenue: 0, flights: 0, revenues: [] }; });
  return state;
};

const selectBanditPrice = (banditState) => {
  // Thompson Sampling: sample from posterior for expected flight revenue
  // Use Gaussian approximation: sample from N(mean_revenue, std_dev / sqrt(n))
  
  let bestPrice = PRICES[0];
  let bestSampledRevenue = -Infinity;
  
  PRICES.forEach(price => {
    const { totalRevenue, flights, revenues } = banditState[price];
    
    if (flights === 0) {
      // Haven't tried this price yet - give it a high optimistic sample
      bestSampledRevenue = Infinity;
      bestPrice = price;
      return;
    }
    
    const meanRevenue = totalRevenue / flights;
    // Calculate std dev of revenues
    const variance = revenues.length > 1 
      ? revenues.reduce((sum, r) => sum + Math.pow(r - meanRevenue, 2), 0) / (revenues.length - 1)
      : 10000; // High variance if only 1 sample
    const stdDev = Math.sqrt(variance);
    
    // Sample from posterior: N(mean, stdDev / sqrt(n))
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const sampledRevenue = meanRevenue + z * stdDev / Math.sqrt(flights);
    
    if (sampledRevenue > bestSampledRevenue) {
      bestSampledRevenue = sampledRevenue;
      bestPrice = price;
    }
  });
  
  return bestPrice;
};

const Seat = ({ status }) => (
  <div style={{
    width: '14px', height: '16px',
    backgroundColor: status === 'available' ? '#2d3748' : status === 'sold-new' ? '#68d391' : '#48bb78',
    borderRadius: '3px 3px 1px 1px', margin: '1px',
    border: status === 'sold-new' ? '2px solid #f6e05e' : '1px solid #4a5568',
    transition: 'all 0.3s ease',
    transform: status === 'sold-new' ? 'scale(1.15)' : 'scale(1)',
  }} />
);

const AirplaneVisualization = ({ seats }) => {
  const rows = [];
  for (let i = 0; i < TOTAL_SEATS; i += 6) rows.push(seats.slice(i, i + 6));
  return (
    <div style={{ background: 'linear-gradient(180deg, #1a202c 0%, #2d3748 100%)', borderRadius: '40px 40px 15px 15px', padding: '20px 15px 15px 15px', position: 'relative', border: '2px solid #4a5568' }}>
      <div style={{ position: 'absolute', top: '-10px', left: '50%', transform: 'translateX(-50%)', width: '40px', height: '20px', background: '#2d3748', borderRadius: '20px 20px 0 0', border: '2px solid #4a5568', borderBottom: 'none' }} />
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1px' }}>
        {rows.map((row, ri) => (
          <div key={ri} style={{ display: 'flex', alignItems: 'center' }}>
            <span style={{ fontSize: '7px', color: '#718096', width: '14px', textAlign: 'right', marginRight: '3px' }}>{ri + 1}</span>
            {row.slice(0, 3).map((seat, si) => <Seat key={si} status={seat} />)}
            <div style={{ width: '6px' }} />
            {row.slice(3).map((seat, si) => <Seat key={si + 3} status={seat} />)}
          </div>
        ))}
      </div>
    </div>
  );
};

export default function AirlinePricingSimulator() {
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [policy, setPolicy] = useState('static');
  const [flightNum, setFlightNum] = useState(0);
  const [currentDay, setCurrentDay] = useState(1);
  const [customerIndex, setCustomerIndex] = useState(0);
  const dailyCustomersRef = useRef(0);
  const [seats, setSeats] = useState(Array(TOTAL_SEATS).fill('available'));
  const [currentPrice, setCurrentPrice] = useState(199);
  const [staticPrice, setStaticPrice] = useState(199);
  const [lastCustomer, setLastCustomer] = useState(null);
  const [flightRevenue, setFlightRevenue] = useState(0);
  const [flightTickets, setFlightTickets] = useState(0);
  const [dayRevenue, setDayRevenue] = useState(0);
  const [revenueHistory, setRevenueHistory] = useState([]);
  const [qParams, setQParams] = useState({ learningRate: 0.3, discount: 0.9, epsilon: 0.1 });
  const [banditExploration, setBanditExploration] = useState(50);
  const [showParams, setShowParams] = useState(false);
  const [policyResults, setPolicyResults] = useState({});
  const maxFlights = 20;
  
  // Tab and optimizer state
  const [activeTab, setActiveTab] = useState('simulator');
  const [optimizerRunning, setOptimizerRunning] = useState(false);
  const [optimizerProgress, setOptimizerProgress] = useState({ generation: 0, totalGenerations: 20 });
  const [optimizerResults, setOptimizerResults] = useState([]);
  const [bestParams, setBestParams] = useState(null);
  const [optimizerConfig, setOptimizerConfig] = useState({ 
    populationSize: 20, 
    generations: 20, 
    flightsPerEval: 100,
    mutationRate: 0.2
  });

  const Q = useRef(initQTable());
  const banditState = useRef(initBanditState());
  const banditFlightPrice = useRef(null);  // Track bandit's chosen price for the flight
  const dayStartState = useRef(null);
  const dayPrice = useRef(null);
  const stopRef = useRef(false);
  const dayRevenueRef = useRef(0);
  const flightHistory = useRef([]);  // Track (state, action) pairs for Q-learning

  const seatsRemaining = seats.filter(s => s === 'available').length;
  const daysLeft = TOTAL_DAYS - currentDay + 1;

  const resetFlight = useCallback(() => {
    setCurrentDay(1);
    setCustomerIndex(0);
    dailyCustomersRef.current = 0;
    setSeats(Array(TOTAL_SEATS).fill('available'));
    setFlightRevenue(0);
    setFlightTickets(0);
    setDayRevenue(0);
    setLastCustomer(null);
    dayStartState.current = null;
    dayPrice.current = null;
    banditFlightPrice.current = null;
    dayRevenueRef.current = 0;
    flightHistory.current = [];
  }, []);

  const fullReset = useCallback(() => {
    setIsRunning(false);
    stopRef.current = true;
    setFlightNum(0);
    setRevenueHistory([]);
    Q.current = initQTable();
    banditState.current = initBanditState();
    resetFlight();
  }, [resetFlight]);

  const selectDailyPrice = useCallback((seatsRem, dLeft) => {
    const state = getState(seatsRem, dLeft);
    if (policy === 'static') return staticPrice;
    if (policy === 'bandit') {
      // For bandit, use the same price for the entire flight
      if (banditFlightPrice.current === null) {
        banditFlightPrice.current = selectBanditPrice(banditState.current);
      }
      return banditFlightPrice.current;
    }
    if (policy === 'qlearning') {
      if (Math.random() < qParams.epsilon) return PRICES[Math.floor(Math.random() * PRICES.length)];
      let bestPrice = PRICES[0], bestQ = -Infinity;
      PRICES.forEach(p => { const qVal = getQValue(Q.current, state, p); if (qVal > bestQ) { bestQ = qVal; bestPrice = p; } });
      return bestPrice;
    }
    return staticPrice;
  }, [policy, staticPrice, qParams.epsilon]);

  const endDay = useCallback((finalDayRevenue, seatsRem, dLeft) => {
    if (policy === 'qlearning' && dayStartState.current !== null && dayPrice.current !== null) {
      // Store state-action pair for end-of-flight update
      flightHistory.current.push({ state: dayStartState.current, action: dayPrice.current });
    }
  }, [policy]);

  const simulationStep = useCallback(() => {
    if (stopRef.current) { setIsRunning(false); return; }
    
    // Stop if we've reached max flights
    if (flightNum >= maxFlights) {
      setIsRunning(false);
      return;
    }

    const seatsRem = seats.filter(s => s === 'available').length;
    const dLeft = TOTAL_DAYS - currentDay + 1;

    if (currentDay > TOTAL_DAYS || seatsRem === 0) {
      // End of flight: update bandit state with flight revenue
      if (policy === 'bandit' && banditFlightPrice.current !== null) {
        const price = banditFlightPrice.current;
        banditState.current[price].totalRevenue += flightRevenue;
        banditState.current[price].flights++;
        banditState.current[price].revenues.push(flightRevenue);
        banditState.current.totalFlights++;
      }
      // End of flight: update Q-values using total flight revenue
      if (policy === 'qlearning' && flightHistory.current.length > 0) {
        const totalRevenue = flightRevenue;
        // Only update each unique state-action pair ONCE per flight
        const visited = new Set();
        const history = flightHistory.current;
        for (let i = 0; i < history.length; i++) {
          const { state, action } = history[i];
          const key = state + '-' + action;
          if (!visited.has(key)) {
            visited.add(key);
            const oldQ = getQValue(Q.current, state, action);
            const newQ = oldQ + qParams.learningRate * (totalRevenue - oldQ);
            setQValue(Q.current, state, action, newQ);
          }
        }
      }
      setRevenueHistory(prev => [...prev, { flight: flightNum + 1, revenue: flightRevenue, tickets: flightTickets, policy, occupancy: Math.round((flightTickets / TOTAL_SEATS) * 100) }]);
      setFlightNum(prev => prev + 1);
      resetFlight();
      return;
    }

    if (customerIndex === 0) {
      // Start of new day - exactly 1 customer per day
      dailyCustomersRef.current = 1;
      const newPrice = selectDailyPrice(seatsRem, dLeft);
      setCurrentPrice(newPrice);
      setDayRevenue(0);
      dayRevenueRef.current = 0;
      dayStartState.current = getState(seatsRem, dLeft);
      dayPrice.current = newPrice;
    }

    if (customerIndex < dailyCustomersRef.current && seatsRem > 0) {
      const wtp = generateCustomerWTP(dLeft);
      const price = dayPrice.current || currentPrice;
      const bought = wtp >= price;
      setLastCustomer({ wtp: wtp.toFixed(0), price, bought });

      if (bought) {
        setSeats(prev => {
          const newSeats = [...prev];
          const idx = newSeats.findIndex(s => s === 'available');
          if (idx !== -1) newSeats[idx] = speed === Infinity ? 'sold' : 'sold-new';
          return newSeats;
        });
        setFlightRevenue(prev => prev + price);
        setFlightTickets(prev => prev + 1);
        setDayRevenue(prev => prev + price);
        dayRevenueRef.current += price;
        // Skip animation at max speed
        if (speed !== Infinity) {
          setTimeout(() => setSeats(curr => curr.map(s => s === 'sold-new' ? 'sold' : s)), 150);
        }
      }
      setCustomerIndex(prev => prev + 1);
    }

    if (customerIndex >= dailyCustomersRef.current) {
      const newSeatsRem = seats.filter(s => s === 'available').length - (seats.some(s => s === 'sold-new') ? 1 : 0);
      endDay(dayRevenueRef.current, newSeatsRem, dLeft);
      setCurrentDay(prev => prev + 1);
      setCustomerIndex(0);
      dailyCustomersRef.current = 0;
    }
  }, [seats, currentDay, customerIndex, flightNum, flightRevenue, flightTickets, currentPrice, policy, selectDailyPrice, endDay, resetFlight, qParams, maxFlights]);

  // Batch simulate multiple flights synchronously (for max speed)
  const batchSimulateFlights = useCallback((numFlights) => {
    const results = [];
    const currentQ = Q.current;
    const currentBandit = banditState.current;
    
    for (let f = 0; f < numFlights; f++) {
      let seatsRem = TOTAL_SEATS;
      let revenue = 0;
      let tickets = 0;
      const history = [];
      
      // For bandit, select price once at start of flight
      const flightPrice = policy === 'bandit' ? selectBanditPrice(currentBandit) : null;
      
      for (let day = 1; day <= TOTAL_DAYS && seatsRem > 0; day++) {
        const dLeft = TOTAL_DAYS - day + 1;
        const state = getState(seatsRem, dLeft);
        
        // Select price based on policy (bandit uses flightPrice set at start)
        let price;
        if (policy === 'static') {
          price = staticPrice;
        } else if (policy === 'bandit') {
          price = flightPrice;
        } else if (policy === 'qlearning') {
          if (Math.random() < qParams.epsilon) {
            price = PRICES[Math.floor(Math.random() * PRICES.length)];
          } else {
            let bestPrice = PRICES[0], bestQ = -Infinity;
            PRICES.forEach(p => {
              const qVal = getQValue(currentQ, state, p);
              if (qVal > bestQ) { bestQ = qVal; bestPrice = p; }
            });
            price = bestPrice;
          }
        } else {
          price = staticPrice;
        }
        
        history.push({ state, action: price });
        
        // Generate 1 customer per day
        const wtp = generateCustomerWTP(dLeft);
        const bought = wtp >= price && seatsRem > 0;
        
        if (bought) {
          revenue += price;
          tickets++;
          seatsRem--;
        }
      }
      
      // Update bandit state at end of flight
      if (policy === 'bandit') {
        currentBandit[flightPrice].totalRevenue += revenue;
        currentBandit[flightPrice].flights++;
        currentBandit[flightPrice].revenues.push(revenue);
        currentBandit.totalFlights++;
      }
      
      // Update Q-table at end of flight
      if (policy === 'qlearning' && history.length > 0) {
        const visited = new Set();
        for (const { state, action } of history) {
          const key = state + '-' + action;
          if (!visited.has(key)) {
            visited.add(key);
            const oldQ = getQValue(currentQ, state, action);
            const newQ = oldQ + qParams.learningRate * (revenue - oldQ);
            setQValue(currentQ, state, action, newQ);
          }
        }
      }
      
      results.push({ 
        flight: flightNum + results.length + 1, 
        revenue, 
        tickets, 
        policy, 
        occupancy: Math.round((tickets / TOTAL_SEATS) * 100) 
      });
    }
    
    return results;
  }, [policy, staticPrice, qParams, flightNum]);

  useEffect(() => {
    if (!isRunning) return;
    stopRef.current = false;
    
    const runStep = () => {
      if (speed === Infinity) {
        // Batch process 100 flights at once
        const flightsRemaining = maxFlights - flightNum;
        if (flightsRemaining <= 0) {
          setIsRunning(false);
          return;
        }
        const batchSize = Math.min(100, flightsRemaining);
        const results = batchSimulateFlights(batchSize);
        setRevenueHistory(prev => [...prev, ...results]);
        setFlightNum(prev => prev + batchSize);
        setSeats(Array(TOTAL_SEATS).fill('available'));
        setCurrentDay(1);
        setFlightRevenue(0);
        setFlightTickets(0);
      } else {
        simulationStep();
      }
    };
    
    const interval = setInterval(runStep, speed === Infinity ? 1 : 120 / speed);
    return () => clearInterval(interval);
  }, [isRunning, speed, simulationStep, batchSimulateFlights, maxFlights, flightNum]);

  // Save policy results when simulation completes
  useEffect(() => {
    const policyData = revenueHistory.filter(r => r.policy === policy);
    if (policyData.length >= maxFlights) {
      const avgRev = policyData.reduce((s, x) => s + x.revenue, 0) / policyData.length;
      const avgOcc = policyData.reduce((s, x) => s + x.occupancy, 0) / policyData.length;
      const last5 = policyData.slice(-5);
      const last5AvgRev = last5.reduce((s, x) => s + x.revenue, 0) / last5.length;
      const last5AvgOcc = last5.reduce((s, x) => s + x.occupancy, 0) / last5.length;
      setPolicyResults(prev => ({
        ...prev,
        [policy]: { 
          flights: policyData.length, 
          avgRevenue: avgRev, 
          avgOccupancy: avgOcc,
          last5AvgRevenue: last5AvgRev,
          last5AvgOccupancy: last5AvgOcc
        }
      }));
    }
  }, [revenueHistory, policy, maxFlights]);

  const getPolicyName = (p) => ({ static: 'Static', bandit: 'Bandit (TS)', qlearning: 'Q-Learning' }[p] || p);
  const getPolicyColor = (p) => ({ static: '#a0aec0', bandit: '#f6ad55', qlearning: '#68d391' }[p] || '#888');
  const getRecentAvg = (n = 10) => { const r = revenueHistory.filter(x => x.policy === policy).slice(-n); return r.length ? r.reduce((s, x) => s + x.revenue, 0) / r.length : 0; };
  const getAllTimeAvg = () => { const r = revenueHistory.filter(x => x.policy === policy); return r.length ? r.reduce((s, x) => s + x.revenue, 0) / r.length : 0; };

  // Generate WTP for a given day and totalDays
  const generateWTP = (day, totalDays) => {
    // Scale so that WTP explosion happens proportionally regardless of totalDays
    // At day = totalDays, we want (day/divisor)^6 to give a good high value
    // Use divisor = totalDays/3 so last day gives (3)^6 = 729
    const divisor = totalDays / 3;
    const meanWTP = 50 + Math.pow(day / divisor, 6);
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return Math.max(0, meanWTP + z * 40);
  };

  // Get time bucket for given days left - MUST match main simulation's getTimeBucket
  const getTimeBucketForDays = (daysLeft, totalDays) => {
    if (daysLeft >= 60) return 'far';
    if (daysLeft >= 15) return 'medium';
    return 'near';
  };

  // Get state for given seats remaining, days left, and totals
  const getStateForConfig = (seatsRem, daysLeft, totalSeats, totalDays) => {
    const capPct = seatsRem / totalSeats;
    const cap = capPct > 0.7 ? 'high' : capPct > 0.3 ? 'medium' : 'low';
    const time = getTimeBucketForDays(daysLeft, totalDays);
    return cap + '-' + time;
  };

  // Simulate a static pricing flight
  const simulateStaticFlight = (staticPrice, totalSeats, totalDays) => {
    let seatsRem = totalSeats;
    let revenue = 0;
    
    for (let day = 1; day <= totalDays && seatsRem > 0; day++) {
      const numCustomers = samplePoisson(ARRIVAL_RATE);
      for (let c = 0; c < numCustomers && seatsRem > 0; c++) {
        const wtp = generateWTP(day, totalDays);
        if (wtp >= staticPrice) {
          revenue += staticPrice;
          seatsRem--;
        }
      }
    }
    return revenue;
  };

  // Evaluate best static price for given config
  const evaluateBestStatic = (totalSeats, totalDays, numFlights) => {
    let bestPrice = PRICES[0];
    let bestRevenue = 0;
    
    for (const price of PRICES) {
      let totalRev = 0;
      for (let f = 0; f < numFlights; f++) {
        totalRev += simulateStaticFlight(price, totalSeats, totalDays);
      }
      const avgRev = totalRev / numFlights;
      if (avgRev > bestRevenue) {
        bestRevenue = avgRev;
        bestPrice = price;
      }
    }
    return { bestPrice, bestRevenue };
  };

  // Simulate a full flight with given Q-learning parameters and config
  const simulateFlightWithConfig = (params, QTable, totalSeats, totalDays) => {
    let seatsRem = totalSeats;
    let revenue = 0;
    const history = [];
    
    for (let day = 1; day <= totalDays && seatsRem > 0; day++) {
      const dLeft = totalDays - day + 1;
      const state = getStateForConfig(seatsRem, dLeft, totalSeats, totalDays);
      
      // Select price (epsilon-greedy)
      let price;
      if (Math.random() < params.epsilon) {
        price = PRICES[Math.floor(Math.random() * PRICES.length)];
      } else {
        let bestPrice = PRICES[0], bestQ = -Infinity;
        PRICES.forEach(p => {
          const qVal = getQValue(QTable, state, p);
          if (qVal > bestQ) { bestQ = qVal; bestPrice = p; }
        });
        price = bestPrice;
      }
      
      history.push({ state, action: price });
      
      // Generate Poisson number of customers
      const numCustomers = samplePoisson(ARRIVAL_RATE);
      for (let c = 0; c < numCustomers && seatsRem > 0; c++) {
        const wtp = generateWTP(day, totalDays);
        if (wtp >= price) {
          revenue += price;
          seatsRem--;
        }
      }
    }
    
    return { revenue, history };
  };

  // Simulate a full flight with given Q-learning parameters (synchronous, no React state)
  const simulateFlight = (params, QTable) => {
    return simulateFlightWithConfig(params, QTable, TOTAL_SEATS, TOTAL_DAYS);
  };

  // Train Q-table and return average revenue for Q-learning
  const evaluateQLearning = (params, totalSeats, totalDays, numFlights) => {
    const QTable = initQTable();
    const revenues = [];
    
    for (let f = 0; f < numFlights; f++) {
      const { revenue, history } = simulateFlightWithConfig(params, QTable, totalSeats, totalDays);
      revenues.push(revenue);
      
      // Only update each unique state-action pair ONCE per flight
      const visited = new Set();
      for (let i = 0; i < history.length; i++) {
        const { state, action } = history[i];
        const key = state + '-' + action;
        if (!visited.has(key)) {
          visited.add(key);
          const oldQ = getQValue(QTable, state, action);
          const newQ = oldQ + params.learningRate * (revenue - oldQ);
          setQValue(QTable, state, action, newQ);
        }
      }
    }
    
    // Return average of last half of flights (after learning)
    const lastHalf = revenues.slice(Math.floor(numFlights / 2));
    return lastHalf.reduce((a, b) => a + b, 0) / lastHalf.length;
  };

  // Train Q-table over multiple flights and return average revenue (original function for compatibility)
  const evaluateParams = (params, numFlights) => {
    return evaluateQLearning(params, TOTAL_SEATS, TOTAL_DAYS, numFlights);
  };

  // Genetic algorithm optimizer - finds config that maximizes Q-learning advantage over static
  const runOptimizer = async () => {
    setOptimizerRunning(true);
    setOptimizerResults([]);
    setBestParams(null);
    
    const { populationSize, generations, flightsPerEval, mutationRate } = optimizerConfig;
    
    // Initialize population with random Q-learning parameters (use fixed TOTAL_SEATS and TOTAL_DAYS)
    let population = Array(populationSize).fill(null).map(() => ({
      learningRate: Math.random() * 0.49 + 0.01,
      discount: Math.random() * 0.49 + 0.5,
      epsilon: Math.random() * 0.4,
      seats: TOTAL_SEATS,   // Use same as main simulation
      days: TOTAL_DAYS       // Use same as main simulation
    }));
    
    const results = [];
    
    for (let gen = 0; gen < generations; gen++) {
      setOptimizerProgress({ generation: gen + 1, totalGenerations: generations });
      
      // Evaluate fitness for each individual (Q-learning revenue - best static revenue)
      const fitness = await new Promise(resolve => {
        setTimeout(() => {
          resolve(population.map(params => {
            // Q-learning: train for flightsPerEval, measure last half
            const qRevenue = evaluateQLearning(params, params.seats, params.days, flightsPerEval);
            // Static: simulate same number of flights as Q-learning's evaluation period (last half)
            const evalFlights = Math.floor(flightsPerEval / 2);
            const { bestRevenue: staticRevenue, bestPrice } = evaluateBestStatic(params.seats, params.days, evalFlights);
            return { 
              advantage: qRevenue - staticRevenue, 
              qRevenue, 
              staticRevenue,
              bestStaticPrice: bestPrice
            };
          }));
        }, 0);
      });
      
      // Sort by advantage (descending)
      const sorted = population.map((p, i) => ({ params: p, ...fitness[i] }))
        .sort((a, b) => b.advantage - a.advantage);
      
      const bestGen = sorted[0];
      results.push({ 
        generation: gen + 1, 
        bestFitness: bestGen.advantage,
        qRevenue: bestGen.qRevenue,
        staticRevenue: bestGen.staticRevenue,
        bestStaticPrice: bestGen.bestStaticPrice,
        avgFitness: fitness.reduce((a, b) => a + b.advantage, 0) / fitness.length,
        bestParams: { ...bestGen.params }
      });
      setOptimizerResults([...results]);
      
      if (gen === generations - 1) {
        setBestParams(bestGen.params);
        break;
      }
      
      // Selection: keep top 50%
      const survivors = sorted.slice(0, Math.floor(populationSize / 2));
      
      // Crossover and mutation to create new population
      const newPopulation = survivors.map(s => ({ ...s.params }));
      
      while (newPopulation.length < populationSize) {
        const parent1 = survivors[Math.floor(Math.random() * survivors.length)].params;
        const parent2 = survivors[Math.floor(Math.random() * survivors.length)].params;
        
        // Crossover
        const child = {
          learningRate: Math.random() < 0.5 ? parent1.learningRate : parent2.learningRate,
          discount: Math.random() < 0.5 ? parent1.discount : parent2.discount,
          epsilon: Math.random() < 0.5 ? parent1.epsilon : parent2.epsilon,
          seats: Math.random() < 0.5 ? parent1.seats : parent2.seats,
          days: Math.random() < 0.5 ? parent1.days : parent2.days
        };
        
        // Mutation (only Q-learning params, seats/days are fixed)
        if (Math.random() < mutationRate) {
          child.learningRate = Math.max(0.01, Math.min(0.5, child.learningRate + (Math.random() - 0.5) * 0.1));
        }
        if (Math.random() < mutationRate) {
          child.discount = Math.max(0.5, Math.min(0.99, child.discount + (Math.random() - 0.5) * 0.1));
        }
        if (Math.random() < mutationRate) {
          child.epsilon = Math.max(0, Math.min(0.4, child.epsilon + (Math.random() - 0.5) * 0.1));
        }
        
        newPopulation.push(child);
      }
      
      population = newPopulation;
    }
    
    setOptimizerRunning(false);
  };

  // Test results state
  const [testResults, setTestResults] = useState(null);

  // Test function to verify optimizer matches main simulation
  const runComparisonTest = () => {
    const testParams = { ...qParams };
    const numFlights = 100;
    
    // Run optimizer's version
    const optimizerRevenue = evaluateQLearning(testParams, TOTAL_SEATS, TOTAL_DAYS, numFlights);
    
    // Run a simplified version matching main simulation logic exactly
    const QTable2 = initQTable();
    const revenues2 = [];
    
    for (let f = 0; f < numFlights; f++) {
      let seatsRem = TOTAL_SEATS;
      let revenue = 0;
      const history = [];
      
      for (let currentDay = 1; currentDay <= TOTAL_DAYS && seatsRem > 0; currentDay++) {
        const dLeft = TOTAL_DAYS - currentDay + 1;
        const state = getState(seatsRem, dLeft);  // Use main simulation's getState
        
        // Select price (same logic as selectDailyPrice)
        let price;
        if (Math.random() < testParams.epsilon) {
          price = PRICES[Math.floor(Math.random() * PRICES.length)];
        } else {
          let bestPrice = PRICES[0], bestQ = -Infinity;
          PRICES.forEach(p => { 
            const qVal = getQValue(QTable2, state, p); 
            if (qVal > bestQ) { bestQ = qVal; bestPrice = p; } 
          });
          price = bestPrice;
        }
        
        history.push({ state, action: price });
        
        // Generate Poisson number of customers (same as main simulation)
        const numCustomers = samplePoisson(ARRIVAL_RATE);
        for (let c = 0; c < numCustomers && seatsRem > 0; c++) {
          const wtp = generateCustomerWTP(dLeft);
          if (wtp >= price) {
            revenue += price;
            seatsRem--;
          }
        }
      }
      
      revenues2.push(revenue);
      
      // Update Q-table (same logic as main simulation)
      const visited = new Set();
      for (let i = 0; i < history.length; i++) {
        const { state, action } = history[i];
        const key = state + '-' + action;
        if (!visited.has(key)) {
          visited.add(key);
          const oldQ = getQValue(QTable2, state, action);
          const newQ = oldQ + testParams.learningRate * (revenue - oldQ);
          setQValue(QTable2, state, action, newQ);
        }
      }
    }
    
    const lastHalf2 = revenues2.slice(Math.floor(numFlights / 2));
    const mainSimRevenue = lastHalf2.reduce((a, b) => a + b, 0) / lastHalf2.length;
    
    setTestResults({
      optimizer: optimizerRevenue.toFixed(0),
      mainSim: mainSimRevenue.toFixed(0),
      params: testParams
    });
  };

  const applyBestParams = () => {
    if (bestParams) {
      setQParams({ 
        learningRate: bestParams.learningRate, 
        discount: bestParams.discount, 
        epsilon: bestParams.epsilon 
      });
      setActiveTab('simulator');
      setPolicy('qlearning');
      fullReset();
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #1a202c 0%, #2d3748 50%, #1a365d 100%)', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif', padding: '16px' }}>
      <div style={{ textAlign: 'center', marginBottom: '16px' }}>
        <p style={{ fontSize: '0.7rem', color: '#718096', letterSpacing: '0.15em', margin: '0 0 6px 0', textTransform: 'uppercase' }}>UCLA Anderson MGMT 298D</p>
        <h1 style={{ fontSize: '1.6rem', fontWeight: '300', margin: 0, background: 'linear-gradient(135deg, #63b3ed, #4fd1c5)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>✈️ Airline Dynamic Pricing</h1>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '12px' }}>
          <button onClick={() => setActiveTab('simulator')} style={{ background: activeTab === 'simulator' ? '#4299e1' : '#2d3748', border: '1px solid #4a5568', color: activeTab === 'simulator' ? '#1a202c' : '#a0aec0', padding: '6px 16px', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer', fontWeight: activeTab === 'simulator' ? '600' : '400' }}>Simulator</button>
          <button onClick={() => setActiveTab('instructions')} style={{ background: activeTab === 'instructions' ? '#4299e1' : '#2d3748', border: '1px solid #4a5568', color: activeTab === 'instructions' ? '#1a202c' : '#a0aec0', padding: '6px 16px', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer', fontWeight: activeTab === 'instructions' ? '600' : '400' }}>Instructions</button>
        </div>
      </div>

      {activeTab === 'instructions' && (
        <div style={{ maxWidth: '800px', margin: '0 auto', background: '#2d3748', borderRadius: '12px', padding: '24px', border: '1px solid #4a5568' }}>
          <h2 style={{ fontSize: '1.2rem', color: '#63b3ed', margin: '0 0 16px 0' }}>📖 How the Simulation Works</h2>
          
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '1rem', color: '#4fd1c5', margin: '0 0 8px 0' }}>Overview</h3>
            <p style={{ fontSize: '0.85rem', color: '#a0aec0', lineHeight: 1.6, margin: 0 }}>
              You are an airline selling seats on a series of flights. Each flight has <strong style={{ color: '#e2e8f0' }}>75 seats</strong> and a <strong style={{ color: '#e2e8f0' }}>180-day</strong> booking window. 
              One customer arrives each day with a willingness-to-pay (WTP) that increases over time — customers booking closer to departure are willing to pay more.
              Your goal is to maximize revenue by choosing the right pricing strategy.
            </p>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '1rem', color: '#4fd1c5', margin: '0 0 8px 0' }}>Sequence of Events (Each Day)</h3>
            <ol style={{ fontSize: '0.85rem', color: '#a0aec0', lineHeight: 1.8, margin: 0, paddingLeft: '20px' }}>
              <li>The pricing policy selects a price for the day</li>
              <li>A customer arrives with a random WTP drawn from: <strong style={{ color: '#f6e05e' }}>WTP ~ N(100 + t + (t/50)⁵, 10²)</strong> where t = day number</li>
              <li>If WTP ≥ price, the customer buys a ticket; otherwise they leave</li>
              <li>The flight ends when all 180 days pass or all seats are sold</li>
              <li>After {maxFlights} flights, compare performance across policies</li>
            </ol>
          </div>

          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '1rem', color: '#4fd1c5', margin: '0 0 8px 0' }}>Pricing Strategies</h3>
            
            <div style={{ background: '#1a202c', borderRadius: '8px', padding: '12px', marginBottom: '10px', borderLeft: '3px solid #a0aec0' }}>
              <h4 style={{ fontSize: '0.9rem', color: '#a0aec0', margin: '0 0 6px 0' }}>📊 Static Pricing</h4>
              <p style={{ fontSize: '0.8rem', color: '#718096', margin: 0 }}>
                Set a fixed price for all customers across all flights. Use the slider to choose your price.
                This is the baseline — can the learning algorithms beat your chosen static price?
              </p>
            </div>

            <div style={{ background: '#1a202c', borderRadius: '8px', padding: '12px', marginBottom: '10px', borderLeft: '3px solid #f6ad55' }}>
              <h4 style={{ fontSize: '0.9rem', color: '#f6ad55', margin: '0 0 6px 0' }}>🎰 Bandit (Thompson Sampling)</h4>
              <p style={{ fontSize: '0.8rem', color: '#718096', margin: 0 }}>
                A multi-armed bandit that learns the best static price through exploration and exploitation.
                It picks one price per flight, observes the total revenue, and updates its beliefs.
                Thompson Sampling balances trying new prices vs. exploiting known good ones using Bayesian posterior sampling.
              </p>
            </div>

            <div style={{ background: '#1a202c', borderRadius: '8px', padding: '12px', borderLeft: '3px solid #68d391' }}>
              <h4 style={{ fontSize: '0.9rem', color: '#68d391', margin: '0 0 6px 0' }}>🧠 Q-Learning</h4>
              <p style={{ fontSize: '0.8rem', color: '#718096', margin: 0 }}>
                A reinforcement learning algorithm that learns to price dynamically based on the current state 
                (capacity remaining × time until departure). Unlike the bandit, Q-learning can learn different 
                optimal prices for different situations — e.g., price low when seats are plentiful and time is far, 
                price high when seats are scarce or departure is near.
              </p>
            </div>
          </div>

          <div>
            <h3 style={{ fontSize: '1rem', color: '#4fd1c5', margin: '0 0 8px 0' }}>Key Insight</h3>
            <p style={{ fontSize: '0.85rem', color: '#a0aec0', lineHeight: 1.6, margin: 0 }}>
              Because customer WTP increases over time, the <strong style={{ color: '#e2e8f0' }}>optimal strategy is dynamic</strong>: 
              price lower early to fill seats, then raise prices as departure approaches. 
              Q-learning can discover this pattern, while the bandit and static policies are limited to finding the best single price.
            </p>
          </div>
        </div>
      )}

      {activeTab === 'simulator' && (
      <>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '12px', flexWrap: 'wrap' }}>
        <select value={policy} onChange={(e) => { setPolicy(e.target.value); fullReset(); }} style={{ background: '#2d3748', border: '2px solid ' + getPolicyColor(policy), color: '#e2e8f0', padding: '6px 12px', borderRadius: '6px', fontSize: '0.8rem' }}>
          <option value="static">📊 Static</option>
          <option value="bandit">🎰 Bandit (Thompson)</option>
          <option value="qlearning">🧠 Q-Learning</option>
        </select>
        <button onClick={() => { stopRef.current = false; setIsRunning(!isRunning); }} style={{ background: isRunning ? 'linear-gradient(135deg, #e53e3e, #c53030)' : 'linear-gradient(135deg, #48bb78, #38a169)', border: 'none', color: 'white', padding: '6px 16px', borderRadius: '6px', fontSize: '0.8rem', fontWeight: '600', cursor: 'pointer' }}>{isRunning ? '⏸ PAUSE' : '▶ RUN'}</button>
        <button onClick={fullReset} style={{ background: '#4a5568', border: '1px solid #718096', color: '#e2e8f0', padding: '6px 12px', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer' }}>↺ RESET</button>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', background: '#2d3748', padding: '4px 10px', borderRadius: '6px', border: '1px solid #4a5568' }}>
          <span style={{ fontSize: '0.7rem', color: '#a0aec0' }}>Speed:</span>
          {[1, 10, 50].map(s => (<button key={s} onClick={() => setSpeed(s)} style={{ background: speed === s ? '#4299e1' : 'transparent', border: 'none', color: speed === s ? '#1a202c' : '#a0aec0', padding: '2px 6px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.75rem' }}>{s}x</button>))}
        </div>
        {policy === 'static' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: '#2d3748', padding: '4px 10px', borderRadius: '6px', border: '1px solid #4a5568' }}>
            <span style={{ fontSize: '0.7rem', color: '#a0aec0' }}>${staticPrice}</span>
            <input type="range" min={0} max={70} step="1" value={staticPrice === 0 ? 0 : Math.round((staticPrice + 1) / 10)} onChange={(e) => { const v = parseInt(e.target.value); setStaticPrice(v === 0 ? 0 : v * 10 - 1); }} disabled={isRunning} style={{ width: '150px' }} />
          </div>
        )}
        {policy === 'qlearning' && <button onClick={() => setShowParams(!showParams)} style={{ background: showParams ? '#4299e1' : '#4a5568', border: '1px solid #718096', color: '#e2e8f0', padding: '6px 10px', borderRadius: '6px', fontSize: '0.8rem', cursor: 'pointer' }}>⚙️</button>}
      </div>

      {showParams && policy === 'qlearning' && (
        <div style={{ maxWidth: '600px', margin: '0 auto 12px', background: '#2d3748', borderRadius: '8px', padding: '12px', border: '1px solid #48bb7840' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
            {[['learningRate', 'α', 0.01, 0.5], ['discount', 'γ', 0.5, 0.99], ['epsilon', 'ε', 0, 0.4]].map(([k, l, min, max]) => (
              <div key={k}><label style={{ fontSize: '0.7rem', color: '#a0aec0' }}>{l}: {qParams[k].toFixed(3)}</label><input type="range" min={min} max={max} step="0.01" value={qParams[k]} onChange={(e) => setQParams(p => ({ ...p, [k]: parseFloat(e.target.value) }))} disabled={isRunning} style={{ width: '100%' }} /></div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: '12px', maxWidth: '950px', margin: '0 auto' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ background: '#2d3748', borderRadius: '10px', padding: '12px', border: '1px solid #4a5568' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.75rem', color: '#a0aec0' }}>Flight #{flightNum + 1}</span>
              <span style={{ fontSize: '0.75rem', color: '#f6e05e', fontWeight: '600' }}>Day {currentDay}/{TOTAL_DAYS}</span>
            </div>
            <AirplaneVisualization seats={seats} />
            <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: '10px', fontSize: '0.7rem' }}>
              <div style={{ textAlign: 'center' }}><div style={{ color: '#718096' }}>Sold</div><div style={{ color: '#48bb78', fontWeight: '600', fontSize: '1rem' }}>{TOTAL_SEATS - seatsRemaining}</div></div>
              <div style={{ textAlign: 'center' }}><div style={{ color: '#718096' }}>Left</div><div style={{ color: '#a0aec0', fontWeight: '600', fontSize: '1rem' }}>{seatsRemaining}</div></div>
            </div>
          </div>
          <div style={{ background: '#2d3748', borderRadius: '10px', padding: '12px', border: '1px solid #4a5568' }}>
            <div style={{ fontSize: '0.6rem', color: '#718096' }}>THIS FLIGHT</div>
            <div style={{ fontSize: '1.4rem', fontWeight: '700', color: '#68d391' }}>${flightRevenue.toLocaleString()}</div>
            <div style={{ fontSize: '0.65rem', color: '#a0aec0' }}>{flightTickets} tix @ ${flightTickets ? (flightRevenue / flightTickets).toFixed(0) : 0} avg</div>
          </div>
          <div style={{ background: '#2d3748', borderRadius: '10px', padding: '10px', border: '1px solid #4a5568' }}>
            <div style={{ fontSize: '0.55rem', color: '#718096' }}>STATE</div>
            <div style={{ fontSize: '0.85rem', fontWeight: '600', color: '#4fd1c5' }}>{getState(seatsRemaining, daysLeft)}</div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
            <div style={{ background: '#2d3748', borderRadius: '8px', padding: '10px', border: '1px solid #4a5568', textAlign: 'center' }}><div style={{ fontSize: '0.5rem', color: '#718096' }}>FLIGHTS</div><div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#4299e1' }}>{revenueHistory.filter(r => r.policy === policy).length}/{maxFlights}</div></div>
            <div style={{ background: '#2d3748', borderRadius: '8px', padding: '10px', border: '1px solid #4a5568', textAlign: 'center' }}><div style={{ fontSize: '0.5rem', color: '#718096' }}>LAST</div><div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#68d391' }}>${getRecentAvg(1).toFixed(0)}</div></div>
            <div style={{ background: '#2d3748', borderRadius: '8px', padding: '10px', border: '1px solid #4a5568', textAlign: 'center' }}><div style={{ fontSize: '0.5rem', color: '#718096' }}>AVG (LAST 5)</div><div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#f6ad55' }}>${getRecentAvg(5).toFixed(0)}</div></div>
            <div style={{ background: '#2d3748', borderRadius: '8px', padding: '10px', border: '1px solid #4a5568', textAlign: 'center' }}><div style={{ fontSize: '0.5rem', color: '#718096' }}>AVG (ALL)</div><div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#b794f4' }}>${getAllTimeAvg().toFixed(0)}</div></div>
            <div style={{ background: '#2d3748', borderRadius: '8px', padding: '10px', border: '1px solid #4a5568', textAlign: 'center' }}><div style={{ fontSize: '0.5rem', color: '#718096' }}>POLICY</div><div style={{ fontSize: '0.8rem', fontWeight: '600', color: getPolicyColor(policy) }}>{getPolicyName(policy)}</div></div>
          </div>

          <div style={{ background: '#2d3748', borderRadius: '10px', padding: '12px', border: '1px solid #4a5568', flex: 1 }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '0.75rem', color: '#a0aec0' }}>CUMULATIVE AVERAGE REVENUE {policy === 'qlearning' && <span style={{ color: '#68d391' }}>— Watch it learn!</span>}</h3>
            <ResponsiveContainer width="100%" height={120}>
              <AreaChart data={revenueHistory.filter(r => r.policy === policy).map((r, i, arr) => ({ ...r, avgRevenue: arr.slice(0, i + 1).reduce((sum, x) => sum + x.revenue, 0) / (i + 1) }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                <XAxis dataKey="flight" tick={{ fill: '#718096', fontSize: 10 }} stroke="#4a5568" />
                <YAxis tick={{ fill: '#718096', fontSize: 10 }} stroke="#4a5568" domain={[10000, 25000]} />
                <Tooltip contentStyle={{ background: '#1a202c', border: '1px solid #4a5568', borderRadius: '6px', fontSize: '11px' }} formatter={(v) => ['$' + v.toLocaleString(), 'Avg Revenue']} />
                <Area type="monotone" dataKey="avgRevenue" stroke={getPolicyColor(policy)} fill={getPolicyColor(policy) + '40'} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div style={{ background: '#2d3748', borderRadius: '10px', padding: '12px', border: '1px solid #4a5568' }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '0.75rem', color: '#a0aec0' }}>REVENUE PER FLIGHT</h3>
            <ResponsiveContainer width="100%" height={100}>
              <AreaChart data={revenueHistory.filter(r => r.policy === policy)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                <XAxis dataKey="flight" tick={{ fill: '#718096', fontSize: 10 }} stroke="#4a5568" />
                <YAxis tick={{ fill: '#718096', fontSize: 10 }} stroke="#4a5568" />
                <Tooltip contentStyle={{ background: '#1a202c', border: '1px solid #4a5568', borderRadius: '6px', fontSize: '11px' }} formatter={(v) => ['$' + v.toLocaleString(), 'Revenue']} />
                <Area type="monotone" dataKey="revenue" stroke={getPolicyColor(policy)} fill={getPolicyColor(policy) + '40'} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div style={{ background: '#2d3748', borderRadius: '10px', padding: '12px', border: '1px solid #4a5568' }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '0.75rem', color: '#a0aec0' }}>ROLLING AVG (LAST 5)</h3>
            <ResponsiveContainer width="100%" height={100}>
              <AreaChart data={revenueHistory.filter(r => r.policy === policy).map((r, i, arr) => ({ ...r, last5Avg: arr.slice(Math.max(0, i - 4), i + 1).reduce((sum, x) => sum + x.revenue, 0) / Math.min(i + 1, 5) }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                <XAxis dataKey="flight" tick={{ fill: '#718096', fontSize: 10 }} stroke="#4a5568" />
                <YAxis tick={{ fill: '#718096', fontSize: 10 }} stroke="#4a5568" />
                <Tooltip contentStyle={{ background: '#1a202c', border: '1px solid #4a5568', borderRadius: '6px', fontSize: '11px' }} formatter={(v) => ['$' + v.toLocaleString(), 'Avg (5)']} />
                <Area type="monotone" dataKey="last5Avg" stroke="#b794f4" fill="#b794f440" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div style={{ background: '#2d3748', borderRadius: '10px', padding: '12px', border: '1px solid #4a5568' }}>
            <h3 style={{ margin: '0 0 8px 0', fontSize: '0.75rem', color: '#a0aec0' }}>OCCUPANCY % BY FLIGHT</h3>
            <ResponsiveContainer width="100%" height={100}>
              <AreaChart data={revenueHistory.filter(r => r.policy === policy)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                <XAxis dataKey="flight" tick={{ fill: '#718096', fontSize: 10 }} stroke="#4a5568" />
                <YAxis tick={{ fill: '#718096', fontSize: 10 }} stroke="#4a5568" domain={[0, 100]} />
                <Tooltip contentStyle={{ background: '#1a202c', border: '1px solid #4a5568', borderRadius: '6px', fontSize: '11px' }} formatter={(v) => [v + '%', 'Occupancy']} />
                <Area type="monotone" dataKey="occupancy" stroke="#f6ad55" fill="#f6ad5540" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {policy === 'qlearning' && (
            <div style={{ background: '#2d3748', borderRadius: '10px', padding: '12px', border: '1px solid #48bb7840' }}>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '0.7rem', color: '#68d391' }}>Q-TABLE: LEARNED BEST PRICE PER STATE</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'auto repeat(3, 1fr)', gap: '3px', fontSize: '0.6rem' }}>
                <div></div>
                {['far', 'medium', 'near'].map(t => <div key={t} style={{ textAlign: 'center', color: '#718096', padding: '3px', fontSize: '0.55rem' }}>{t === 'far' ? '≥60d' : t === 'medium' ? '15-59d' : '<15d'}</div>)}
                {['high', 'medium', 'low'].map(cap => (
                  <React.Fragment key={cap}>
                    <div style={{ color: '#718096', padding: '3px', fontSize: '0.55rem' }}>{cap === 'high' ? '>70%' : cap === 'medium' ? '30-70%' : '<30%'}</div>
                    {['far', 'medium', 'near'].map(time => {
                      const state = cap + '-' + time;
                      let bestPrice = PRICES[0], bestQ = -Infinity;
                      PRICES.forEach(p => { const q = getQValue(Q.current, state, p); if (q > bestQ) { bestQ = q; bestPrice = p; } });
                      const isCurrent = getState(seatsRemaining, daysLeft) === state;
                      return <div key={time} style={{ textAlign: 'center', padding: '5px 3px', background: isCurrent ? '#2d4a2d' : '#1a202c', borderRadius: '3px', color: '#f6e05e', fontWeight: '600', border: isCurrent ? '2px solid #68d391' : 'none' }}>${bestPrice}</div>;
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>
          )}

          {policy === 'bandit' && (
            <div style={{ background: '#2d3748', borderRadius: '10px', padding: '12px', border: '1px solid #f6ad5540' }}>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '0.7rem', color: '#f6ad55' }}>THOMPSON SAMPLING: AVG FLIGHT REVENUE</h3>
              <div style={{ display: 'flex', justifyContent: 'space-around', fontSize: '0.65rem', flexWrap: 'wrap', gap: '4px' }}>
                {PRICES.map(p => { const d = banditState.current[p]; return (<div key={p} style={{ textAlign: 'center', minWidth: '45px' }}><div style={{ color: '#718096' }}>${p}</div><div style={{ color: '#f6ad55', fontWeight: '600' }}>${d.flights ? (d.totalRevenue / d.flights).toFixed(0) : '?'}</div><div style={{ color: '#4a5568', fontSize: '0.5rem' }}>n={d.flights}</div></div>); })}
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={{ maxWidth: '950px', margin: '12px auto 0', background: '#2d374880', borderRadius: '8px', padding: '12px', border: '1px solid #4a5568' }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: '0.75rem', color: '#a0aec0' }}>📊 Policy Comparison</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #4a5568' }}>
              <th style={{ padding: '8px', textAlign: 'left', color: '#a0aec0' }}>Policy</th>
              <th style={{ padding: '8px', textAlign: 'right', color: '#a0aec0' }}>Flights</th>
              <th style={{ padding: '8px', textAlign: 'right', color: '#a0aec0' }}>Avg Revenue</th>
              <th style={{ padding: '8px', textAlign: 'right', color: '#a0aec0' }}>Avg Occ</th>
              <th style={{ padding: '8px', textAlign: 'right', color: '#b794f4' }}>Last 5 Rev</th>
              <th style={{ padding: '8px', textAlign: 'right', color: '#b794f4' }}>Last 5 Occ</th>
            </tr>
          </thead>
          <tbody>
            {['static', 'bandit', 'qlearning'].map(p => {
              const result = policyResults[p];
              return (
                <tr key={p} style={{ borderBottom: '1px solid #4a556840' }}>
                  <td style={{ padding: '8px', color: getPolicyColor(p), fontWeight: '600' }}>{getPolicyName(p)}</td>
                  <td style={{ padding: '8px', textAlign: 'right', color: '#e2e8f0' }}>{result ? result.flights : '—'}</td>
                  <td style={{ padding: '8px', textAlign: 'right', color: '#68d391', fontWeight: '600' }}>{result ? '$' + result.avgRevenue.toFixed(0) : '—'}</td>
                  <td style={{ padding: '8px', textAlign: 'right', color: '#f6ad55' }}>{result ? result.avgOccupancy.toFixed(1) + '%' : '—'}</td>
                  <td style={{ padding: '8px', textAlign: 'right', color: '#b794f4', fontWeight: '600' }}>{result ? '$' + result.last5AvgRevenue.toFixed(0) : '—'}</td>
                  <td style={{ padding: '8px', textAlign: 'right', color: '#b794f4' }}>{result ? result.last5AvgOccupancy.toFixed(1) + '%' : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      </>
      )}
    </div>
  );
}

