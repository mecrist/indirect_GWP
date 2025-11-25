import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict


#%%%%%%%% FILTERING REACTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Check if reaction involves nitrogen
def has_nitrogen(reaction: str) -> bool:
    return re.search(r'N', reaction, re.IGNORECASE) is not None


#%%%%%%%%%%%%%%% PLOTTING REACTION RATES DICT%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_reaction_rates(reaction_rates_dict):
    for reaction_name, rate_data in reaction_rates_dict.items():
        # Dictionary to store data separated by concentration
        curves = {'10': [], '40': [], '60': []}
        
        # Extract temperature and concentration from the rate data
        for (temp, conc), rate in rate_data.items():
            conc_str = str(conc)
            if conc_str in curves:
                curves[conc_str].append((temp, rate))
        
        # Sort each curve by temperature
        for conc in curves:
            curves[conc].sort(key=lambda x: x[0])
        
        # Plotting :)
        if any(curves[conc] for conc in curves):
            plt.figure(figsize=(10, 6))
            
            for conc, points in curves.items():
                if points:  # only plot if there are data points
                    temps, rates = zip(*points)
                    plt.semilogy(temps, rates, marker='o', linewidth=2, markersize=8, label=f'{conc} CH₄')
            
            plt.xlabel('Temperature (K)', fontsize=20)
            plt.ylabel('Reaction Rate (cm^3/mol)s$^{-1$}', fontsize=20)
            plt.title(f'Reaction Rate vs Temperature\n{reaction_name}', fontsize=20)
            plt.legend(fontsize=15)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            #plt.savefig(f'reaction_rate{reaction_name}.png', dpi=300)
            
            plt.show()

#%%%%%%%%%%%%% EXTRACTING UNIQUE REPRESENTATION FOR THE REACTIONS AND LATEX FOR PLOTTING %%%%%%%%%%%%%%%%%%%%
# RE expression to extract atoms from species strings like [H][C]([H])[N]=[C]
_atom_re = re.compile(r'\[([^]]+)\]')

def extract_composition(species_str):
    """Return a dict element and count from a species string like [H][C]([H])[N]=[C]"""
    atoms = _atom_re.findall(species_str)
    comp = {}
    for a in atoms:
        comp[a] = comp.get(a, 0) + 1
    return comp

def canonical_key_from_comp(comp):
    """Return a canonical, hashable key for a composition dict"""
    # sort alphabetically
    return tuple(sorted(comp.items()))

def canonical_key(species_str):
    """Integrates the extract_composition and canonical_key_from_comp"""
    return canonical_key_from_comp(extract_composition(species_str))

def formula_label_from_key(key):
    """
    Convert canonical key ( (('C',1),('O',2)) ) in a LaTeX label string like r'$CO_2$'.
    Chemical ordering: prefer C, H, then O, N, others alphabetical. Used for plotting.
    """
    order = ['C','H','O','N','S','P']  # common order, then remaining alphabetical
    items = list(key)
    remaining = sorted([e for e,c in items if e not in order])
    ordered = []
    for el in order + remaining:
        for e,c in items:
            if e == el:
                ordered.append((e,c))
    if not ordered:
        ordered = items
    parts = []
    for e,c in ordered:
        if c == 1:
            parts.append(f"{e}")
        else:
            parts.append(f"{e}_{{{c}}}")
    return r"$" + "".join(parts) + r"$"

def parse_reaction_string(reaction_str):
    """
    Parse a reaction string into reactants and products.
    Format: '[reactant1]+[reactant2]+...->[product1]+[product2]+...'
    """
    if '->' not in reaction_str:
        raise ValueError(f"Invalid reaction string: {reaction_str}")
    
    reactants_str, products_str = reaction_str.split('->', 1)
    
    # Split by '+' but be careful of nested structures
    reactants = [s.strip() for s in reactants_str.split('+') if s.strip()]
    products = [s.strip() for s in products_str.split('+') if s.strip()]
    
    return reactants, products

def canonical_reaction_key(reaction_str):
    """
    Create a canonical key for a reaction by normalizing reactants and products.
    This handles different representations of the same chemical reaction.
    """
    reactants, products = parse_reaction_string(reaction_str)
    
    # Convert each species to canonical composition key
    reactant_keys = [canonical_key(r) for r in reactants]
    product_keys = [canonical_key(p) for p in products]
    
    # Sort reactants and products to make order-independent
    # Also sort by canonical key to ensure consistent ordering
    sorted_reactants = tuple(sorted(reactant_keys))
    sorted_products = tuple(sorted(product_keys))
    
    return (sorted_reactants, sorted_products)

def reaction_label_from_key(reaction_key):
    """
    Convert canonical reaction key to a readable label.
    Format: "reactant1 + reactant2 → product1 + product2"
    """
    reactants, products = reaction_key
    
    def format_side(species_list):
        formatted = []
        for species_key in species_list:
            # Convert canonical key back to formula
            formula = formula_label_from_key(species_key)
            formatted.append(formula)
        return " + ".join(formatted)
    
    reactants_str = format_side(reactants)
    products_str = format_side(products)
    
    return f"{reactants_str} → {products_str}"

def build_reaction_groups(reaction_rates_dict):
    """
    Group equivalent reaction representations by their canonical chemical structure.
    
    Returns:
        groups: dict {canonical_reaction_key: list_of_equivalent_reaction_strings}
    """
    groups = defaultdict(list)
    
    for reaction_str in reaction_rates_dict.keys():
        try:
            canonical_key = canonical_reaction_key(reaction_str)
            groups[canonical_key].append(reaction_str)
        except (ValueError, Exception) as e:
            print(f"Warning: Could not parse reaction '{reaction_str}': {e}")
            continue
    
    return dict(groups)

def aggregate_reaction_rates(reaction_rates_dict):
    """
    Aggregate reaction rates for equivalent chemical reactions.
    Combines data from different string representations of the same reaction.
    """
    groups = build_reaction_groups(reaction_rates_dict)
    aggregated_rates = {}
    
    for canonical_key, reaction_strings in groups.items():
        # Combine all rate data for equivalent reactions
        combined_rates = {}
        
        for reaction_str in reaction_strings:
            rate_data = reaction_rates_dict[reaction_str]
            for (temp, conc), rate in rate_data.items():
                # If multiple representations have data for same (temp, conc),
                # we'll keep the first one encountered
                if (temp, conc) not in combined_rates:
                    combined_rates[(temp, conc)] = rate
                else:
                    # Option: you could average or handle duplicates differently
                    print(f"Warning: Duplicate data for {canonical_key} at T={temp}, conc={conc}")
        
        if combined_rates:
            aggregated_rates[canonical_key] = {
                'rate_data': combined_rates,
                'original_strings': reaction_strings,
                'label': reaction_label_from_key(canonical_key)
            }
    
    return aggregated_rates

def plot_aggregated_reactions(aggregated_rates):
    """
    Plot the aggregated reaction rates grouped by canonical reaction.
    Also prints a formatted table of temperature, CH₄ concentration, and reaction rate.
    """
    for reaction_key, data in aggregated_rates.items():
        rate_data = data['rate_data']
        label = data['label']
        
        # Print a formatted table of values
        print(f"\n--- {label} ---")
        print("Temp (K)\tCH₄ Conc\tRate (cm³/mol·s⁻¹)")
        print("-" * 45)
        for (temp, conc), rate in sorted(rate_data.items()):
            print(f"{temp:<10}\t{conc:<10}\t{rate:.3e}")
        
        # Dictionary to store data separated by concentration
        curves = {'10': [], '40': [], '60': []}
        
        # Extract temperature and concentration from the rate data
        for (temp, conc), rate in rate_data.items():
            conc_str = str(conc)
            if conc_str in curves:
                curves[conc_str].append((temp, rate))
        
        # Sort each curve by temperature
        for conc in curves:
            curves[conc].sort(key=lambda x: x[0])
        
        # Create plot only if there's data
        if any(curves[conc] for conc in curves):
            plt.figure(figsize=(10, 6))
            
            for conc, points in curves.items():
                if points:  # only plot if there are data points
                    temps, rates = zip(*points)
                    plt.semilogy(temps, rates, marker='o', linewidth=2, markersize=8, 
                                label=f'{conc} CH₄')
            
            plt.xlabel('Temperatura (K)', fontsize=15)
            plt.ylabel('Constante de velocidade (cm³/mol s$^{-1}$)', fontsize=15)
            plt.title(f'Constante de velocidade de reação vs T \n{label}', fontsize=20)
            plt.tick_params(axis='both', labelsize=14)
            plt.legend(fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'reaction_rate_{label}.png', dpi=300)
            plt.show()


# Defyning integrated usage
def analyze_reaction_equivalences(reaction_rates_dict):
    """
    Analyze and print information about equivalent reactions.
    """
    groups = build_reaction_groups(reaction_rates_dict)
    
    print("Reaction Group Analysis:")
    print("=" * 80)
    
    for canonical_key, reaction_strings in groups.items():
        if len(reaction_strings) > 1:
            print(f"\nEquivalent representations found ({len(reaction_strings)} variants):")
            print(f"Canonical reaction: {reaction_label_from_key(canonical_key)}")
            print("Original strings:")
            for i, r_str in enumerate(reaction_strings, 1):
                print(f"  {i}. {r_str}")
    
    return groups