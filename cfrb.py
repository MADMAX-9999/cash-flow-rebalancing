import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random

# Konfiguracja strony
st.set_page_config(
    page_title="Symulator Inwestycji w Metale Szlachetne - LBMA",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stałe przeliczeniowe
TROY_OUNCE_TO_GRAMS = 31.1035  # 1 uncja trojańska = 31.1035 gramów

@st.cache_data
def load_lbma_data():
    """Wczytaj i przygotuj dane LBMA"""
    try:
        # Wczytanie danych z pliku CSV
        df = pd.read_csv('lbma_data.csv')
        
        # Konwersja dat - upewnij się, że są w formacie datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Usuń rekordy z nieprawidłowymi datami
        df = df.dropna(subset=['Date'])
        
        # Konwersja z uncji na gramy (podzielenie przez wagę uncji trojańskiej)
        df['Gold_EUR_per_gram'] = df['Gold_EUR'] / TROY_OUNCE_TO_GRAMS
        df['Silver_EUR_per_gram'] = df['Silver_EUR'] / TROY_OUNCE_TO_GRAMS
        df['Platinum_EUR_per_gram'] = df['Platinum_EUR'] / TROY_OUNCE_TO_GRAMS
        df['Palladium_EUR_per_gram'] = df['Palladium_EUR'] / TROY_OUNCE_TO_GRAMS
        
        # Posortowanie według daty
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    except FileNotFoundError:
        st.error("❌ Nie znaleziono pliku lbma_data.csv. Upewnij się, że plik znajduje się w tym samym folderze co aplikacja.")
        return None
    except Exception as e:
        st.error(f"❌ Błąd podczas wczytywania danych: {str(e)}")
        return None

def calculate_monthly_returns(df, start_date, months):
    """Oblicz miesięczne zmiany cen na podstawie rzeczywistych danych LBMA"""
    
    try:
        # Konwersja start_date na datetime - różne przypadki
        if pd.api.types.is_datetime64_any_dtype(start_date):
            start_date_dt = start_date
        elif isinstance(start_date, str):
            start_date_dt = pd.to_datetime(start_date)
        elif hasattr(start_date, 'date'):  # obiekt date z Streamlit
            start_date_dt = pd.to_datetime(start_date)
        else:
            # Konwersja przez string
            start_date_dt = pd.to_datetime(str(start_date))
        
        # Upewnij się, że kolumna Date jest w formacie datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
        
        # Konwersja na ten sam timezone (usuń timezone jeśli istnieje)
        if hasattr(start_date_dt, 'tz') and start_date_dt.tz is not None:
            start_date_dt = start_date_dt.tz_localize(None)
        
        # Usuń timezone z kolumny Date jeśli istnieje
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dtype.tz is not None:
            df = df.copy()
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        # Filtrowanie danych od podanej daty
        df_filtered = df[df['Date'] >= start_date_dt].copy()
        
    except Exception as e:
        st.error(f"Błąd w calculate_monthly_returns: {str(e)}")
        return generate_simulated_returns(months), pd.DataFrame()
    
    if len(df_filtered) == 0:
        st.warning(f"⚠️ Brak danych od {start_date.strftime('%Y-%m-%d')}. Używam najnowszych dostępnych danych.")
        df_filtered = df.tail(months * 30).copy()  # Przybliżenie: 30 dni = 1 miesiąc
    
    # Grupowanie po miesiącach i obliczanie średnich cen
    df_filtered['YearMonth'] = df_filtered['Date'].dt.to_period('M')
    monthly_avg = df_filtered.groupby('YearMonth')[
        ['Gold_EUR_per_gram', 'Silver_EUR_per_gram', 'Platinum_EUR_per_gram', 'Palladium_EUR_per_gram']
    ].mean().reset_index()
    
    if len(monthly_avg) < 2:
        st.warning("⚠️ Za mało danych historycznych. Używam symulowanych zmian.")
        return generate_simulated_returns(months)
    
    # Obliczenie miesięcznych zwrotów
    returns = {}
    metals = ['Gold', 'Silver', 'Platinum', 'Palladium']
    
    for metal in metals:
        col_name = f'{metal}_EUR_per_gram'
        if col_name in monthly_avg.columns:
            prices = monthly_avg[col_name].values
            monthly_changes = []
            
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i-1]) / prices[i-1]
                monthly_changes.append(change)
            
            # Jeśli potrzebujemy więcej danych niż mamy, uzupełniamy losowo z historycznych
            while len(monthly_changes) < months:
                monthly_changes.extend(monthly_changes[:min(len(monthly_changes), months - len(monthly_changes))])
            
            returns[metal.lower()] = monthly_changes[:months]
    
    return returns, monthly_avg

def generate_simulated_returns(months):
    """Generuj symulowane zwroty jako fallback"""
    np.random.seed(42)  # Dla powtarzalności
    return {
        'gold': np.random.normal(0.008, 0.025, months).tolist(),
        'silver': np.random.normal(0.005, 0.035, months).tolist(),
        'platinum': np.random.normal(0.003, 0.030, months).tolist(),
        'palladium': np.random.normal(0.010, 0.045, months).tolist()
    }

def get_current_prices(df):
    """Pobierz najnowsze ceny metali za gram"""
    if df is None or len(df) == 0:
        return None
    
    latest_data = df.iloc[-1]
    return {
        'gold': latest_data['Gold_EUR_per_gram'],
        'silver': latest_data['Silver_EUR_per_gram'],
        'platinum': latest_data['Platinum_EUR_per_gram'],
        'palladium': latest_data['Palladium_EUR_per_gram'],
        'date': latest_data['Date'].strftime('%Y-%m-%d')
    }

# Tytuł aplikacji
st.title("🥇 Symulator Inwestycji w Metale Szlachetne")
st.markdown("### Oparte na rzeczywistych danych LBMA (ceny za gram w EUR)")

# Wczytanie danych
with st.spinner('📊 Wczytywanie danych LBMA...'):
    lbma_df = load_lbma_data()

if lbma_df is not None:
    current_prices = get_current_prices(lbma_df)
    
    # Wyświetlenie aktualnych cen
    if current_prices:
        st.sidebar.markdown("### 💰 Aktualne ceny (EUR/gram)")
        st.sidebar.markdown(f"**Data:** {current_prices['date']}")
        st.sidebar.markdown(f"**🥇 Złoto:** {current_prices['gold']:.2f} €/g")
        st.sidebar.markdown(f"**🥈 Srebro:** {current_prices['silver']:.3f} €/g")
        st.sidebar.markdown(f"**⚪ Platyna:** {current_prices['platinum']:.2f} €/g")
        st.sidebar.markdown(f"**⚫ Pallad:** {current_prices['palladium']:.2f} €/g")
        st.sidebar.markdown("---")

# Sidebar - parametry symulacji
st.sidebar.header("⚙️ Parametry Symulacji")

initial_investment = st.sidebar.number_input(
    "Kapitał początkowy (EUR)", 
    min_value=100, 
    max_value=1000000, 
    value=10000, 
    step=500
)

monthly_contribution = st.sidebar.number_input(
    "Miesięczna wpłata (EUR)", 
    min_value=0, 
    max_value=10000, 
    value=500, 
    step=50
)

simulation_months = st.sidebar.slider(
    "Okres symulacji (miesiące)", 
    min_value=6, 
    max_value=60, 
    value=24, 
    step=6
)

# Wybór okresu startowego dla danych historycznych
if lbma_df is not None and len(lbma_df) > 0:
    try:
        min_date = lbma_df['Date'].min().date()
        max_date = lbma_df['Date'].max().date()
        
        start_date = st.sidebar.date_input(
            "Data rozpoczęcia symulacji",
            value=max_date - timedelta(days=365*2),  # 2 lata wstecz
            min_value=min_date,
            max_value=max_date
        )
    except Exception as e:
        st.sidebar.error(f"Błąd z datami: {str(e)}")
        start_date = None
else:
    start_date = None

rebalance_frequency = st.sidebar.selectbox(
    "Częstotliwość rebalansingu",
    options=[1, 3, 6, 12],
    index=1,
    format_func=lambda x: f"Co {x} miesiąc(e/y)"
)

st.sidebar.subheader("🎯 Docelowa alokacja (%)")

# Docelowa alokacja
target_gold = st.sidebar.slider("🥇 Złoto (%)", 0, 100, 40, 5)
target_silver = st.sidebar.slider("🥈 Srebro (%)", 0, 100, 30, 5)
target_platinum = st.sidebar.slider("⚪ Platyna (%)", 0, 100, 20, 5)
target_palladium = st.sidebar.slider("⚫ Pallad (%)", 0, 100, 10, 5)

# Sprawdzenie czy suma wynosi 100%
total_allocation = target_gold + target_silver + target_platinum + target_palladium
if total_allocation != 100:
    st.sidebar.error(f"⚠️ Suma alokacji: {total_allocation}%. Musi wynosić 100%!")

# Przycisk uruchomienia symulacji
run_simulation = st.sidebar.button("🚀 Uruchom Symulację", type="primary")

def run_portfolio_simulation(initial_inv, monthly_cont, months, rebalance_freq, allocations, price_changes):
    """Główna funkcja symulacji portfela z rzeczywistymi danymi"""
    
    # Inicjalizacja portfela w gramach
    if current_prices:
        initial_grams = {
            'gold': (initial_inv * allocations['gold'] / 100) / current_prices['gold'],
            'silver': (initial_inv * allocations['silver'] / 100) / current_prices['silver'],
            'platinum': (initial_inv * allocations['platinum'] / 100) / current_prices['platinum'],
            'palladium': (initial_inv * allocations['palladium'] / 100) / current_prices['palladium']
        }
    else:
        # Fallback - symulowane ceny startowe
        initial_grams = {
            'gold': (initial_inv * allocations['gold'] / 100) / 75,  # ~75 EUR/gram dla złota
            'silver': (initial_inv * allocations['silver'] / 100) / 0.95,  # ~0.95 EUR/gram dla srebra
            'platinum': (initial_inv * allocations['platinum'] / 100) / 27,  # ~27 EUR/gram dla platyny
            'palladium': (initial_inv * allocations['palladium'] / 100) / 26   # ~26 EUR/gram dla palladu
        }
    
    # Symulowane ceny startowe (aktualne ceny lub fallback)
    prices = current_prices.copy() if current_prices else {
        'gold': 75.0, 'silver': 0.95, 'platinum': 27.0, 'palladium': 26.0
    }
    
    simulation_data = []
    portfolio_grams = initial_grams.copy()
    
    for month in range(months + 1):
        # Zastosuj zmiany cen (oprócz pierwszego miesiąca)
        if month > 0 and month <= len(price_changes['gold']):
            for metal in prices.keys():
                if metal in price_changes and (month-1) < len(price_changes[metal]):
                    change = price_changes[metal][month-1]
                    prices[metal] *= (1 + change)
            
            # Rebalansing co określoną liczbę miesięcy
            if month % rebalance_freq == 0:
                total_contribution = monthly_cont * rebalance_freq
                
                # Oblicz aktualną wartość portfela
                current_values = {
                    metal: portfolio_grams[metal] * prices[metal] 
                    for metal in portfolio_grams.keys()
                }
                current_total = sum(current_values.values())
                
                # Oblicz aktualną alokację
                current_allocation = {
                    metal: (value / current_total) * 100 
                    for metal, value in current_values.items()
                }
                
                # Cash-flow rebalancing - dodaj więcej gramów do niedowartościowanych metali
                for metal in portfolio_grams.keys():
                    current_percent = current_allocation[metal]
                    target_percent = allocations[metal]
                    difference = target_percent - current_percent
                    
                    # Bazowa wpłata + bonus za rebalansing
                    base_addition = total_contribution * (target_percent / 100)
                    rebalance_bonus = max(0, (difference / 100) * total_contribution * 0.5)
                    
                    total_eur_addition = base_addition + rebalance_bonus
                    grams_to_add = total_eur_addition / prices[metal]
                    
                    portfolio_grams[metal] += grams_to_add
        
        # Oblicz wartości w EUR
        current_values = {
            metal: portfolio_grams[metal] * prices[metal] 
            for metal in portfolio_grams.keys()
        }
        total_value = sum(current_values.values())
        
        # Zapisz dane miesiąca
        month_data = {
            'Miesiąc': month,
            'Łączna_wartość': total_value,
            'Złoto_EUR': current_values['gold'],
            'Srebro_EUR': current_values['silver'],
            'Platyna_EUR': current_values['platinum'],
            'Pallad_EUR': current_values['palladium'],
            'Złoto_gramy': portfolio_grams['gold'],
            'Srebro_gramy': portfolio_grams['silver'],
            'Platyna_gramy': portfolio_grams['platinum'],
            'Pallad_gramy': portfolio_grams['palladium'],
            'Cena_złoto': prices['gold'],
            'Cena_srebro': prices['silver'],
            'Cena_platyna': prices['platinum'],
            'Cena_pallad': prices['palladium'],
            'Złoto_%': (current_values['gold'] / total_value) * 100,
            'Srebro_%': (current_values['silver'] / total_value) * 100,
            'Platyna_%': (current_values['platinum'] / total_value) * 100,
            'Pallad_%': (current_values['palladium'] / total_value) * 100
        }
        simulation_data.append(month_data)
    
    return pd.DataFrame(simulation_data), portfolio_grams, prices

# Uruchomienie symulacji
if run_simulation and total_allocation == 100 and lbma_df is not None and start_date is not None:
    with st.spinner('🔄 Analizowanie danych LBMA i symulowanie inwestycji...'):
        
        target_allocations = {
            'gold': target_gold,
            'silver': target_silver,
            'platinum': target_platinum,
            'palladium': target_palladium
        }
        
        # Oblicz miesięczne zmiany cen na podstawie rzeczywistych danych
        price_changes, monthly_avg = calculate_monthly_returns(lbma_df, start_date, simulation_months)
        
        # Uruchom symulację
        df, final_grams, final_prices = run_portfolio_simulation(
            initial_investment,
            monthly_contribution,
            simulation_months,
            rebalance_frequency,
            target_allocations,
            price_changes
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
                f"{final_value:,.0f} €",
                f"{total_return:+,.0f} €"
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
                f"{total_invested:,.0f} €"
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
                line=dict(color='orange', width=3),
                hovertemplate='Miesiąc: %{x}<br>Wartość: €%{y:,.0f}<extra></extra>'
            ))
            
            # Linie dla poszczególnych metali
            colors = {'Złoto_EUR': 'gold', 'Srebro_EUR': 'silver', 'Platyna_EUR': 'lightgray', 'Pallad_EUR': 'lightsteelblue'}
            names = {'Złoto_EUR': 'Złoto', 'Srebro_EUR': 'Srebro', 'Platyna_EUR': 'Platyna', 'Pallad_EUR': 'Pallad'}
            
            for metal, color in colors.items():
                fig.add_trace(go.Scatter(
                    x=df['Miesiąc'], 
                    y=df[metal],
                    mode='lines',
                    name=names[metal],
                    line=dict(color=color, width=2),
                    hovertemplate=f'{names[metal]}: €%{{y:,.0f}}<extra></extra>'
                ))
            
            fig.update_layout(
                xaxis_title="Miesiąc",
                yaxis_title="Wartość (EUR)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🥧 Końcowa alokacja")
            
            # Wykres kołowy
            final_values = [df['Złoto_EUR'].iloc[-1], df['Srebro_EUR'].iloc[-1], 
                          df['Platyna_EUR'].iloc[-1], df['Pallad_EUR'].iloc[-1]]
            labels = ['🥇 Złoto', '🥈 Srebro', '⚪ Platyna', '⚫ Pallad']
            colors_pie = ['gold', 'silver', 'lightgray', 'lightsteelblue']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, 
                values=final_values,
                marker_colors=colors_pie,
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='%{label}<br>Wartość: €%{value:,.0f}<br>Udział: %{percent}<extra></extra>'
            )])
            
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabela szczegółowa portfela
        st.subheader("📋 Szczegóły końcowego portfela")
        
        final_data = df.iloc[-1]
        portfolio_details = pd.DataFrame({
            'Metal': ['🥇 Złoto', '🥈 Srebro', '⚪ Platyna', '⚫ Pallad'],
            'Gramy': [final_grams[metal] for metal in ['gold', 'silver', 'platinum', 'palladium']],
            'Cena za gram (€)': [final_prices[metal] for metal in ['gold', 'silver', 'platinum', 'palladium']],
            'Wartość (€)': [final_data['Złoto_EUR'], final_data['Srebro_EUR'], 
                           final_data['Platyna_EUR'], final_data['Pallad_EUR']],
            'Alokacja (%)': [final_data['Złoto_%'], final_data['Srebro_%'], 
                            final_data['Platyna_%'], final_data['Pallad_%']],
            'Docelowa (%)': [target_gold, target_silver, target_platinum, target_palladium]
        })
        
        portfolio_details['Różnica (%)'] = portfolio_details['Alokacja (%)'] - portfolio_details['Docelowa (%)']
        
        # Formatowanie kolumn
        portfolio_details['Gramy'] = portfolio_details['Gramy'].round(2)
        portfolio_details['Cena za gram (€)'] = portfolio_details['Cena za gram (€)'].round(3)
        portfolio_details['Wartość (€)'] = portfolio_details['Wartość (€)'].round(0)
        portfolio_details['Alokacja (%)'] = portfolio_details['Alokacja (%)'].round(1)
        portfolio_details['Różnica (%)'] = portfolio_details['Różnica (%)'].round(1)
        
        st.dataframe(portfolio_details, use_container_width=True, hide_index=True)
        
        # Wykres cen metali w czasie symulacji
        st.subheader("💹 Zmiany cen metali podczas symulacji")
        
        fig_prices = go.Figure()
        
        price_cols = [('Cena_złoto', '🥇 Złoto', 'gold'), 
                     ('Cena_srebro', '🥈 Srebro', 'silver'),
                     ('Cena_platyna', '⚪ Platyna', 'lightgray'), 
                     ('Cena_pallad', '⚫ Pallad', 'lightsteelblue')]
        
        for col, name, color in price_cols:
            fig_prices.add_trace(go.Scatter(
                x=df['Miesiąc'],
                y=df[col],
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=f'{name}: €%{{y:.3f}}/g<extra></extra>'
            ))
        
        fig_prices.update_layout(
            xaxis_title="Miesiąc",
            yaxis_title="Cena (EUR/gram)",
            hovermode='x unified',
            height=300
        )
        
        st.plotly_chart(fig_prices, use_container_width=True)

elif run_simulation and total_allocation != 100:
    st.error("❌ Nie można uruchomić symulacji - suma alokacji musi wynosić 100%!")
elif run_simulation and lbma_df is None:
    st.error("❌ Nie można uruchomić symulacji - problem z wczytaniem danych LBMA!")
elif run_simulation and start_date is None:
    st.error("❌ Nie można uruchomić symulacji - problem z wyborem daty startowej!")

# Informacje o aplikacji
st.markdown("---")
with st.expander("ℹ️ O danych LBMA"):
    if lbma_df is not None:
        st.markdown(f"""
        **Dane wykorzystane w symulacji:**
        - 📊 **Źródło:** London Bullion Market Association (LBMA)
        - 📅 **Zakres dat:** {lbma_df['Date'].min().strftime('%Y-%m-%d')} do {lbma_df['Date'].max().strftime('%Y-%m-%d')}
        - 📈 **Liczba rekordów:** {len(lbma_df):,}
        - ⚖️ **Jednostka:** Gramy (przeliczone z uncji trojańskich)
        - 💰 **Waluta:** EUR
        
        **Przeliczenie:** 1 uncja trojańska = {TROY_OUNCE_TO_GRAMS} gramów
        """)

with st.expander("🔄 Jak działa rebalansing metodą cash-flow?"):
    st.markdown("""
    **Rebalansing metodą cash-flow** w kontekście metali szlachetnych:
    
    1. **Kupno fizycznych gramów** - każda wpłata kupuje rzeczywiste gramy metali
    2. **Brak sprzedaży** - nie sprzedajemy posiadanych gramów
    3. **Inteligentny zakup** - nowe środki kierujemy proporcjonalnie więcej do metali poniżej docelowej alokacji
    4. **Wykorzystanie zmienności cen** - automatycznie kupujemy więcej gdy ceny spadają
    5. **Minimalne koszty** - brak kosztów sprzedaży i podatków od zysków kapitałowych
    
    **Przykład:** Jeśli złoto ma stanowić 40% portfela, ale aktualnie stanowi 35%, większa część 
    nowych wpłat zostanie przeznaczona na zakup gramów złota.
    """)

# Footer
st.markdown("---")
st.markdown("*Aplikacja wykorzystuje rzeczywiste dane historyczne LBMA • Ceny w EUR za gram*")
