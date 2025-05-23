import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Konfiguracja strony
st.set_page_config(
    page_title="Symulator Inwestycji w Metale Szlachetne",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tytuł aplikacji
st.title("📈 Symulator Inwestycji w Metale Szlachetne")
st.markdown("### Symulacja portfela z rebalansowaniem metodą cash-flow")

# Sidebar - parametry symulacji
st.sidebar.header("⚙️ Parametry Symulacji")

initial_investment = st.sidebar.number_input(
    "Kapitał początkowy (PLN)", 
    min_value=1000, 
    max_value=1000000, 
    value=10000, 
    step=1000
)

monthly_contribution = st.sidebar.number_input(
    "Miesięczna wpłata (PLN)", 
    min_value=0, 
    max_value=10000, 
    value=500, 
    step=100
)

simulation_months = st.sidebar.slider(
    "Okres symulacji (miesiące)", 
    min_value=6, 
    max_value=120, 
    value=24, 
    step=6
)

rebalance_frequency = st.sidebar.selectbox(
    "Częstotliwość rebalansingu",
    options=[1, 3, 6, 12],
    index=1,
    format_func=lambda x: f"Co {x} miesiąc(e/y)"
)

st.sidebar.subheader("🎯 Docelowa alokacja (%)")

# Docelowa alokacja
target_gold = st.sidebar.slider("Złoto (%)", 0, 100, 40, 5)
target_silver = st.sidebar.slider("Srebro (%)", 0, 100, 30, 5)
target_platinum = st.sidebar.slider("Platyna (%)", 0, 100, 20, 5)
target_palladium = st.sidebar.slider("Pallad (%)", 0, 100, 10, 5)

# Sprawdzenie czy suma wynosi 100%
total_allocation = target_gold + target_silver + target_platinum + target_palladium
if total_allocation != 100:
    st.sidebar.error(f"⚠️ Suma alokacji: {total_allocation}%. Musi wynosić 100%!")

# Przycisk uruchomienia symulacji
run_simulation = st.sidebar.button("🚀 Uruchom Symulację", type="primary")

# Dane symulacyjne - historyczne zmiany cen (% miesięcznie)
@st.cache_data
def get_metal_price_changes():
    return {
        'gold': [0.02, -0.01, 0.03, 0.015, -0.02, 0.025, 0.01, -0.015, 0.02, 0.005, 0.03, -0.01],
        'silver': [0.04, -0.03, 0.05, 0.02, -0.04, 0.035, 0.015, -0.025, 0.03, 0.01, 0.045, -0.02],
        'platinum': [0.015, -0.02, 0.025, 0.01, -0.015, 0.02, 0.005, -0.01, 0.015, 0.002, 0.025, -0.005],
        'palladium': [0.06, -0.05, 0.08, 0.03, -0.06, 0.05, 0.02, -0.04, 0.045, 0.015, 0.07, -0.03]
    }

def run_portfolio_simulation(initial_inv, monthly_cont, months, rebalance_freq, allocations):
    """Główna funkcja symulacji portfela"""
    
    metal_changes = get_metal_price_changes()
    
    # Inicjalizacja portfela
    portfolio = {
        'gold': initial_inv * (allocations['gold'] / 100),
        'silver': initial_inv * (allocations['silver'] / 100),
        'platinum': initial_inv * (allocations['platinum'] / 100),
        'palladium': initial_inv * (allocations['palladium'] / 100)
    }
    
    simulation_data = []
    
    for month in range(months + 1):
        # Zastosuj zmiany cen (oprócz pierwszego miesiąca)
        if month > 0:
            for metal in portfolio.keys():
                change_index = (month - 1) % len(metal_changes[metal])
                price_change = metal_changes[metal][change_index]
                portfolio[metal] *= (1 + price_change)
            
            # Rebalansing co określoną liczbę miesięcy
            if month % rebalance_freq == 0:
                total_contribution = monthly_cont * rebalance_freq
                current_total = sum(portfolio.values())
                
                # Oblicz aktualną alokację
                current_allocation = {
                    metal: (value / current_total) * 100 
                    for metal, value in portfolio.items()
                }
                
                # Cash-flow rebalancing
                for metal in portfolio.keys():
                    current_percent = current_allocation[metal]
                    target_percent = allocations[metal]
                    difference = target_percent - current_percent
                    
                    # Bazowa wpłata + bonus za rebalansing
                    base_addition = total_contribution * (target_percent / 100)
                    rebalance_bonus = max(0, difference / 100 * total_contribution * 0.5)
                    
                    portfolio[metal] += base_addition + rebalance_bonus
        
        # Zapisz dane miesiąca
        total_value = sum(portfolio.values())
        month_data = {
            'Miesiąc': month,
            'Łączna_wartość': total_value,
            'Złoto': portfolio['gold'],
            'Srebro': portfolio['silver'],
            'Platyna': portfolio['platinum'],
            'Pallad': portfolio['palladium'],
            'Złoto_%': (portfolio['gold'] / total_value) * 100,
            'Srebro_%': (portfolio['silver'] / total_value) * 100,
            'Platyna_%': (portfolio['platinum'] / total_value) * 100,
            'Pallad_%': (portfolio['palladium'] / total_value) * 100
        }
        simulation_data.append(month_data)
    
    return pd.DataFrame(simulation_data), portfolio

# Uruchomienie symulacji
if run_simulation and total_allocation == 100:
    with st.spinner('🔄 Symulowanie inwestycji...'):
        time.sleep(1)  # Symulacja ładowania
        
        target_allocations = {
            'gold': target_gold,
            'silver': target_silver,
            'platinum': target_platinum,
            'palladium': target_palladium
        }
        
        df, final_portfolio = run_portfolio_simulation(
            initial_investment,
            monthly_contribution,
            simulation_months,
            rebalance_frequency,
            target_allocations
        )
        
        # Obliczenia finansowe
        total_invested = initial_investment + (monthly_contribution * simulation_months)
        final_value = df['Łączna_wartość'].iloc[-1]
        total_return = final_value - total_invested
        return_percentage = (total_return / total_invested) * 100 if total_invested > 0 else 0
        
        # Wyświetlenie statystyk
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Wartość końcowa", 
                f"{final_value:,.0f} zł",
                f"{total_return:+,.0f} zł"
            )
        
        with col2:
            st.metric(
                "📊 Zwrot", 
                f"{return_percentage:+.1f}%",
                f"{'📈' if return_percentage >= 0 else '📉'}"
            )
        
        with col3:
            st.metric(
                "💵 Zainwestowano", 
                f"{total_invested:,.0f} zł"
            )
        
        with col4:
            st.metric(
                "📅 Okres", 
                f"{simulation_months} miesięcy",
                f"{simulation_months/12:.1f} lat"
            )
        
        # Wykresy
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Rozwój wartości portfela w czasie")
            
            fig = go.Figure()
            
            # Linia łącznej wartości
            fig.add_trace(go.Scatter(
                x=df['Miesiąc'], 
                y=df['Łączna_wartość'],
                mode='lines',
                name='Łączna wartość',
                line=dict(color='orange', width=3)
            ))
            
            # Linie dla poszczególnych metali
            colors = {'Złoto': 'gold', 'Srebro': 'silver', 'Platyna': 'lightgray', 'Pallad': 'lightsteelblue'}
            for metal, color in colors.items():
                fig.add_trace(go.Scatter(
                    x=df['Miesiąc'], 
                    y=df[metal],
                    mode='lines',
                    name=metal,
                    line=dict(color=color, width=2)
                ))
            
            fig.update_layout(
                xaxis_title="Miesiąc",
                yaxis_title="Wartość (PLN)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🥧 Końcowa alokacja")
            
            # Wykres kołowy
            final_values = [final_portfolio[metal] for metal in ['gold', 'silver', 'platinum', 'palladium']]
            labels = ['Złoto', 'Srebro', 'Platyna', 'Pallad']
            colors_pie = ['gold', 'silver', 'lightgray', 'lightsteelblue']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, 
                values=final_values,
                marker_colors=colors_pie,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabela szczegółowa
        st.subheader("📋 Szczegóły końcowego portfela")
        
        portfolio_details = pd.DataFrame({
            'Metal': ['Złoto', 'Srebro', 'Platyna', 'Pallad'],
            'Wartość (PLN)': [final_portfolio[metal] for metal in ['gold', 'silver', 'platinum', 'palladium']],
            'Alokacja (%)': [
                (final_portfolio[metal] / final_value) * 100 
                for metal in ['gold', 'silver', 'platinum', 'palladium']
            ],
            'Docelowa (%)': [target_gold, target_silver, target_platinum, target_palladium]
        })
        
        portfolio_details['Różnica (%)'] = portfolio_details['Alokacja (%)'] - portfolio_details['Docelowa (%)']
        portfolio_details['Wartość (PLN)'] = portfolio_details['Wartość (PLN)'].round(0)
        portfolio_details['Alokacja (%)'] = portfolio_details['Alokacja (%)'].round(1)
        portfolio_details['Różnica (%)'] = portfolio_details['Różnica (%)'].round(1)
        
        st.dataframe(portfolio_details, use_container_width=True, hide_index=True)
        
        # Wykres alokacji w czasie
        st.subheader("📊 Zmiana alokacji w czasie")
        
        fig_allocation = go.Figure()
        
        for metal, color in zip(['Złoto_%', 'Srebro_%', 'Platyna_%', 'Pallad_%'], 
                               ['gold', 'silver', 'lightgray', 'lightsteelblue']):
            fig_allocation.add_trace(go.Scatter(
                x=df['Miesiąc'],
                y=df[metal],
                mode='lines',
                name=metal.replace('_%', ''),
                line=dict(color=color, width=2),
                stackgroup='one'
            ))
        
        fig_allocation.update_layout(
            xaxis_title="Miesiąc",
            yaxis_title="Alokacja (%)",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            height=300
        )
        
        st.plotly_chart(fig_allocation, use_container_width=True)

elif run_simulation and total_allocation != 100:
    st.error("❌ Nie można uruchomić symulacji - suma alokacji musi wynosić 100%!")

# Informacje o aplikacji
st.markdown("---")
with st.expander("ℹ️ Jak działa rebalansing metodą cash-flow?"):
    st.markdown("""
    **Rebalansing metodą cash-flow** to strategia, która:
    
    1. **Nie wymaga sprzedaży** istniejących pozycji
    2. **Nowe środki** (miesięczne wpłaty) kieruje proporcjonalnie więcej do metali poniżej docelowej alokacji
    3. **Minimalizuje koszty** transakcyjne i podatki
    4. **Stopniowo przywraca** równowagę portfela
    
    **Przykład:** Jeśli złoto ma mieć 40% portfela, ale aktualnie ma 35%, to większa część nowych wpłat 
    trafi do złota, aby zbliżyć się do docelowej proporcji.
    """)

with st.expander("📈 Informacje o symulacji"):
    st.markdown("""
    **Dane wykorzystane w symulacji:**
    - Historyczne wzorce zmienności metali szlachetnych
    - Symulowane miesięczne zmiany cen
    - Różna charakterystyka ryzyka dla każdego metalu
    
    **Uwaga:** To jest symulacja edukacyjna. Rzeczywiste wyniki mogą się różnić!
    """)

# Footer
st.markdown("---")
st.markdown("*Aplikacja stworzona w Streamlit dla analizy inwestycji w metale szlachetne*")
