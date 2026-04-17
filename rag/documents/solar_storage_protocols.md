# Solar Energy Storage Protocols and Battery Management

## 1. Battery State of Charge (SoC) Management

### 1.1 Optimal SoC Operating Range
Lithium-ion battery systems should maintain state of charge between 20% and 80% during normal operations to maximize cycle life. Deep discharge below 10% SoC should be avoided except during grid emergency conditions. Operating within this range extends battery life by 2-3× compared to full-range cycling.

### 1.2 SoC Targets by Grid Condition
- **Normal Operations:** Maintain SoC at 50% ± 10% to provide symmetric charge/discharge capacity.
- **High Solar Forecast:** Target SoC of 30-40% at sunrise to maximize solar absorption.
- **Low Solar Forecast / Storm Warning:** Charge to 80-90% SoC before the expected generation deficit.
- **Grid Emergency:** SoC limits may be extended to 5-95% for up to 4 hours.

### 1.3 Calendar Aging Mitigation
Batteries stored at high SoC (>80%) experience accelerated calendar aging. During extended periods of low demand, discharge to 40-50% SoC. Temperature-compensated SoC limits should be applied: reduce maximum SoC by 5% for every 10°C above 25°C ambient.

## 2. Charging Strategies

### 2.1 Solar-Optimized Charging
- **Morning Ramp Charging:** Begin charging at reduced rate (C/4) during sunrise ramp to avoid grid stress. Transition to full rate (C/2) once solar output exceeds 50% of rated capacity.
- **Peak Solar Charging:** During peak solar hours (typically 10:00-14:00), charge at maximum rate (C/2 to 1C) to capture excess generation that would otherwise be curtailed.
- **Afternoon Taper:** Reduce charging rate after 14:00 to reserve capacity for evening peak demand response.

### 2.2 Grid-Aware Charging
Charging from the grid should only occur during off-peak hours (typically 22:00-06:00) when electricity prices are lowest. Time-of-use rate arbitrage can offset 15-30% of battery system operating costs. Charging should be suspended when grid frequency drops below 49.8 Hz / 59.8 Hz.

### 2.3 Charging Rate Limits
Maximum charging rate should not exceed the manufacturer's recommended C-rate. For most utility-scale lithium-ion systems: continuous charging at C/2, peak charging at 1C for durations not exceeding 30 minutes. Temperature-dependent derating: reduce charging rate by 20% when cell temperature exceeds 35°C.

## 3. Discharging Strategies

### 3.1 Peak Demand Discharge
Deploy stored energy during evening peak hours (typically 17:00-21:00) when solar generation has ceased and demand remains high. Target discharge to reduce peak demand by 20-40%. Discharge rate should be matched to the demand shortfall to maximize revenue.

### 3.2 Solar Smoothing Discharge
When cloud transients cause solar output drops exceeding 30% in less than 5 minutes, batteries should automatically discharge to smooth the combined output. Response time must be under 200 milliseconds. Discharge rate should match the solar deficit up to the battery's maximum continuous discharge rate.

### 3.3 Emergency Grid Support
During grid contingency events (loss of major generator, transmission fault), batteries should provide maximum discharge within 1 second. This primary frequency response should continue for up to 15 minutes or until conventional reserves are fully deployed. Compensation for emergency services: $200-500/MWh.

### 3.4 Discharge Depth Limits
Regular discharge should not reduce SoC below 20%. For energy arbitrage operations, minimum SoC of 15% is acceptable. Emergency operations may discharge to 5% SoC, but such events should not exceed 50 occurrences per year to maintain warranty compliance.

## 4. Hybrid Solar-Storage Scheduling

### 4.1 Day-Ahead Storage Scheduling
Based on day-ahead solar forecasts and price signals:
1. Calculate expected solar surplus/deficit for each hour
2. Schedule charging during surplus hours, prioritizing hours with lowest grid prices
3. Schedule discharging during deficit hours, prioritizing hours with highest grid prices
4. Maintain 20% SoC reserve for unplanned events

### 4.2 Real-Time Dispatch Optimization
Real-time storage dispatch should consider: (a) current SoC, (b) updated solar forecast, (c) real-time grid price, (d) expected demand for the next 4 hours. Use rolling optimization with 15-minute decision intervals and 4-hour look-ahead horizon.

### 4.3 Seasonal Strategy Adjustments
- **Summer:** Prioritize evening peak discharge (higher AC demand). Increase daily cycling.
- **Winter:** Prioritize morning and evening discharge. Reduce depth of discharge to protect batteries in cold conditions.
- **Shoulder Seasons:** Focus on price arbitrage. Reduce cycling frequency to extend battery life.

## 5. Battery Health Monitoring

### 5.1 Key Performance Indicators
- Round-trip efficiency: should remain above 85%. Investigate if drops below 82%.
- Capacity fade: maximum 2.5% per year for warranted performance. 
- Internal resistance growth: maximum 20% increase over 5 years.
- Cell voltage imbalance: must not exceed 50 mV within a module.

### 5.2 Preventive Maintenance Schedule
- Daily: automated SoC and voltage balance checks
- Weekly: round-trip efficiency calculation
- Monthly: capacity test at C/5 rate
- Quarterly: thermal imaging of connections and inverters
- Annually: full capacity test and impedance spectroscopy
