const csvPath = "data/life_expectancy.csv";
const numberFields = [
  "Year", "Adult_Mortality", "infant_deaths", "Alcohol", "percentage_expenditure",
  "Hepatitis_B", "Measles", "BMI", "under_five_deaths", "Polio", "Total_expenditure",
  "Diphtheria", "HIV_AIDS", "GDP", "Population", "thinness_1_19_years",
  "thinness_5_9_years", "Income_composition_of_resources", "Schooling", "Life_expectancy"
];

const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
let records = [];

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",").map((header) => header.trim());
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => {
      const value = values[index]?.trim() ?? "";
      return [header, value === "" || value.toLowerCase() === "null" ? null : Number(value)];
    }));
  }).filter((row) => Number.isFinite(row.Year));
}

function average(rows, field) {
  const values = rows.map((row) => row[field]).filter(Number.isFinite);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function svgFrame(width, height, left, right, top, bottom) {
  return { width, height, left, right, top, bottom, plotWidth: width - left - right, plotHeight: height - top - bottom };
}

function chartAxes(frame, ticks, format = (value) => numberFormat.format(value)) {
  const { left, top, plotWidth, plotHeight } = frame;
  return ticks.map((tick) => {
    const y = top + plotHeight * tick.position;
    return `<line class="gridline" x1="${left}" y1="${y}" x2="${left + plotWidth}" y2="${y}"/><text class="axis-label" x="${left - 8}" y="${y + 3}" text-anchor="end">${format(tick.value)}</text>`;
  }).join("");
}

function svgWrap(width, height, content, label) {
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}" preserveAspectRatio="none">${content}</svg>`;
}

function renderTrend(rows) {
  const target = document.querySelector("#trendChart");
  if (!rows.length) return showEmpty(target);
  const frame = svgFrame(640, 190, 42, 12, 10, 27);
  const values = rows.map((row) => average(row.items, "Life_expectancy")).filter(Number.isFinite);
  if (!values.length) return showEmpty(target);
  const min = Math.floor(Math.min(...values) / 5) * 5;
  const max = Math.ceil(Math.max(...values) / 5) * 5 || min + 5;
  const yFor = (value) => frame.top + frame.plotHeight - ((value - min) / (max - min || 1)) * frame.plotHeight;
  const xFor = (index) => frame.left + (rows.length === 1 ? frame.plotWidth / 2 : index * frame.plotWidth / (rows.length - 1));
  const path = rows.map((row, index) => `${index ? "L" : "M"}${xFor(index)},${yFor(average(row.items, "Life_expectancy"))}`).join(" ");
  const area = `${path} L${xFor(rows.length - 1)},${frame.top + frame.plotHeight} L${xFor(0)},${frame.top + frame.plotHeight} Z`;
  const ticks = [0, 0.5, 1].map((position) => ({ position, value: max - position * (max - min) }));
  const labels = rows.map((row, index) => index % Math.max(1, Math.ceil(rows.length / 8)) === 0 || index === rows.length - 1
    ? `<text class="axis-label" x="${xFor(index)}" y="${frame.height - 5}" text-anchor="middle">${row.year}</text>` : "").join("");
  const points = rows.map((row, index) => `<circle class="trend-point interactive-point" cx="${xFor(index)}" cy="${yFor(average(row.items, "Life_expectancy"))}" r="3.5"><title>${row.year}: ${numberFormat.format(average(row.items, "Life_expectancy"))} years</title></circle>`).join("");
  target.innerHTML = svgWrap(frame.width, frame.height, `${chartAxes(frame, ticks, (value) => `${Math.round(value)}y`)}<line class="axis-base" x1="${frame.left}" y1="${frame.top + frame.plotHeight}" x2="${frame.left + frame.plotWidth}" y2="${frame.top + frame.plotHeight}"/><path class="line-area" d="${area}"/><path class="trend-line" d="${path}"/>${points}${labels}`, "Average life expectancy trend by year");
}

function renderBars(rows, targetSelector, field, colorClass, formatter = (value) => numberFormat.format(value)) {
  const target = document.querySelector(targetSelector);
  const available = rows.map((row) => ({ ...row, value: average(row.items, field) })).filter((row) => Number.isFinite(row.value));
  if (!available.length) return showEmpty(target);
  const frame = svgFrame(420, 190, 35, 8, 10, 27);
  const max = Math.max(...available.map((row) => row.value)) * 1.12 || 1;
  const step = frame.plotWidth / available.length;
  const barWidth = Math.max(2, step * 0.62);
  const bars = available.map((row, index) => {
    const height = row.value / max * frame.plotHeight;
    const x = frame.left + index * step + (step - barWidth) / 2;
    const y = frame.top + frame.plotHeight - height;
    const label = index % Math.max(1, Math.ceil(available.length / 8)) === 0 || index === available.length - 1
      ? `<text class="axis-label" x="${x + barWidth / 2}" y="${frame.height - 5}" text-anchor="middle">${row.year}</text>` : "";
    return `<rect class="bar ${colorClass}" x="${x}" y="${y}" width="${barWidth}" height="${height}" rx="2"><title>${row.year}: ${formatter(row.value)}</title></rect>${label}`;
  }).join("");
  const ticks = [0, 0.5, 1].map((position) => ({ position, value: max * (1 - position) }));
  target.innerHTML = svgWrap(frame.width, frame.height, `${chartAxes(frame, ticks)}<line class="axis-base" x1="${frame.left}" y1="${frame.top + frame.plotHeight}" x2="${frame.left + frame.plotWidth}" y2="${frame.top + frame.plotHeight}"/>${bars}`, `Annual average ${field.replaceAll("_", " ")}`);
}

function renderCoverage(rows) {
  const target = document.querySelector("#coverageChart");
  const available = rows.map((row) => ({
    ...row,
    expenditure: average(row.items, "Total_expenditure"),
    coverage: average(row.items, "Polio")
  })).filter((row) => Number.isFinite(row.expenditure) || Number.isFinite(row.coverage));
  if (!available.length) return showEmpty(target);
  const frame = svgFrame(520, 175, 36, 8, 9, 25);
  const max = Math.max(10, ...available.map((row) => Math.max(row.expenditure || 0, row.coverage || 0))) * 1.08;
  const step = frame.plotWidth / available.length;
  const width = Math.max(1.5, Math.min(9, step * .28));
  const bars = available.map((row, index) => {
    const center = frame.left + index * step + step / 2;
    const makeBar = (value, x, klass, name) => {
      if (!Number.isFinite(value)) return "";
      const barHeight = value / max * frame.plotHeight;
      return `<rect class="bar ${klass}" x="${x}" y="${frame.top + frame.plotHeight - barHeight}" width="${width}" height="${barHeight}" rx="1"><title>${row.year} ${name}: ${numberFormat.format(value)}</title></rect>`;
    };
    const label = index % Math.max(1, Math.ceil(available.length / 7)) === 0 || index === available.length - 1
      ? `<text class="axis-label" x="${center}" y="${frame.height - 4}" text-anchor="middle">${row.year}</text>` : "";
    return `${makeBar(row.expenditure, center - width - 1, "yellow", "spending")}${makeBar(row.coverage, center + 1, "bar-secondary", "polio coverage")}${label}`;
  }).join("");
  target.innerHTML = svgWrap(frame.width, frame.height, `${chartAxes(frame, [0, .5, 1].map((position) => ({ position, value: max * (1 - position) })))}<line class="axis-base" x1="${frame.left}" y1="${frame.top + frame.plotHeight}" x2="${frame.left + frame.plotWidth}" y2="${frame.top + frame.plotHeight}"/>${bars}`, "Health expenditure and polio immunization coverage by year");
}

function renderScatter(rows) {
  const target = document.querySelector("#scatterChart");
  const points = rows.flatMap((row) => row.items).filter((row) => Number.isFinite(row.Schooling) && Number.isFinite(row.Life_expectancy));
  if (!points.length) return showEmpty(target);
  const frame = svgFrame(520, 175, 42, 9, 9, 26);
  const xMax = Math.ceil(Math.max(...points.map((row) => row.Schooling)) / 5) * 5;
  const yMin = Math.floor(Math.min(...points.map((row) => row.Life_expectancy)) / 10) * 10;
  const yMax = Math.ceil(Math.max(...points.map((row) => row.Life_expectancy)) / 10) * 10;
  const xFor = (value) => frame.left + value / xMax * frame.plotWidth;
  const yFor = (value) => frame.top + frame.plotHeight - (value - yMin) / (yMax - yMin || 1) * frame.plotHeight;
  const ticks = [0, .5, 1].map((position) => ({ position, value: yMax - position * (yMax - yMin) }));
  const xTicks = [0, .5, 1].map((position) => `<text class="axis-label" x="${frame.left + position * frame.plotWidth}" y="${frame.height - 4}" text-anchor="middle">${Math.round(xMax * position)}</text>`).join("");
  const dots = points.map((row) => `<circle class="scatter-point" cx="${xFor(row.Schooling)}" cy="${yFor(row.Life_expectancy)}" r="2.4"><title>Schooling: ${numberFormat.format(row.Schooling)} years; life expectancy: ${numberFormat.format(row.Life_expectancy)} years</title></circle>`).join("");
  target.innerHTML = svgWrap(frame.width, frame.height, `${chartAxes(frame, ticks, (value) => `${Math.round(value)}y`)}<line class="axis-base" x1="${frame.left}" y1="${frame.top + frame.plotHeight}" x2="${frame.left + frame.plotWidth}" y2="${frame.top + frame.plotHeight}"/>${dots}${xTicks}<text class="axis-label" x="${frame.left + frame.plotWidth / 2}" y="${frame.height}" text-anchor="middle">Schooling (years)</text>`, "Scatter plot of schooling and life expectancy observations");
}

function showEmpty(target) {
  target.innerHTML = "<div class=\"empty-state\">No numeric observations for this period</div>";
}

function getFilteredRows() {
  const start = Number(document.querySelector("#yearStart").value);
  const end = Number(document.querySelector("#yearEnd").value);
  return records.filter((row) => row.Year >= start && row.Year <= end);
}

function render() {
  const filtered = getFilteredRows();
  const years = [...new Set(filtered.map((row) => row.Year))].sort((a, b) => a - b);
  const yearly = years.map((year) => ({ year, items: filtered.filter((row) => row.Year === year) }));
  const life = average(filtered, "Life_expectancy");
  const schooling = average(filtered, "Schooling");
  const mortality = average(filtered, "Adult_Mortality");
  const expenditure = average(filtered, "Total_expenditure");
  document.querySelector("#lifeValue").textContent = life === null ? "—" : numberFormat.format(life);
  document.querySelector("#schoolValue").textContent = schooling === null ? "—" : numberFormat.format(schooling);
  document.querySelector("#mortalityValue").textContent = mortality === null ? "—" : numberFormat.format(mortality);
  document.querySelector("#expenditureValue").textContent = expenditure === null ? "—" : numberFormat.format(expenditure);
  document.querySelector("#recordCount").textContent = numberFormat.format(filtered.length);
  document.querySelector("#periodCaption").textContent = `${filtered.length.toLocaleString()} observations in selected period`;
  renderTrend(yearly);
  renderBars(yearly, "#schoolChart", "Schooling", "coral");
  renderCoverage(yearly);
  renderScatter(yearly);
}

function initializeFilters() {
  const years = [...new Set(records.map((row) => row.Year))].sort((a, b) => a - b);
  const start = document.querySelector("#yearStart");
  const end = document.querySelector("#yearEnd");
  for (const year of years) {
    start.add(new Option(year, year));
    end.add(new Option(year, year));
  }
  start.value = years[0];
  end.value = years.at(-1);
  start.addEventListener("change", () => {
    if (Number(start.value) > Number(end.value)) end.value = start.value;
    render();
  });
  end.addEventListener("change", () => {
    if (Number(end.value) < Number(start.value)) start.value = end.value;
    render();
  });
}

function exportSummary() {
  const filtered = getFilteredRows();
  const headers = ["Year", ...numberFields.filter((field) => field !== "Year")];
  const years = [...new Set(filtered.map((row) => row.Year))].sort((a, b) => a - b);
  const lines = [headers.join(","), ...years.map((year) => {
    const group = filtered.filter((row) => row.Year === year);
    return headers.map((field) => field === "Year" ? year : average(group, field) ?? "").join(",");
  })];
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  link.download = "life-expectancy-summary.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

async function loadDashboard() {
  try {
    const response = await fetch(csvPath);
    if (!response.ok) throw new Error(`Could not load ${csvPath} (${response.status})`);
    records = parseCsv(await response.text());
    if (!records.length) throw new Error("The CSV did not contain usable yearly records.");
    initializeFilters();
    render();
    document.querySelector("#exportButton").addEventListener("click", exportSummary);
  } catch (error) {
    document.querySelector("#periodCaption").textContent = "Could not load the local CSV";
    document.querySelectorAll(".chart-wrap").forEach((target) => {
      target.innerHTML = `<div class="empty-state">${error.message}. Open this folder through a local web server.</div>`;
    });
    console.error(error);
  }
}

loadDashboard();