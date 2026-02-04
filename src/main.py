"""
Test simple pour vérifier que toutes les dépendances fonctionnent
"""

print("="*70)
print("TEST DE L'ENVIRONNEMENT")
print("="*70)

# Test 1 : Python version
print("\n1. Version Python:")
import sys
print(f"   ✓ Python {sys.version}")

# Test 2 : NumPy
print("\n2. NumPy:")
try:
    import numpy as np
    print(f"   ✓ NumPy version {np.__version__}")
    # Test rapide
    arr = np.array([1, 2, 3])
    print(f"   ✓ Opération test: sum([1,2,3]) = {np.sum(arr)}")
except ImportError as e:
    print(f"   ✗ ERREUR: {e}")

# Test 3 : MOSEK
print("\n3. MOSEK:")
try:
    import mosek
    print(f"   ✓ MOSEK importé avec succès")
    
    # Test création d'un modèle simple
    from mosek.fusion import Model, Domain
    with Model("test") as M:
        x = M.variable("x", 1, Domain.greaterThan(0.0))
        print(f"   ✓ Modèle MOSEK créé avec succès")
    
except ImportError as e:
    print(f"   ✗ ERREUR d'import: {e}")
except Exception as e:
    print(f"   ✗ ERREUR MOSEK: {e}")
    print(f"   → Vérifiez votre licence MOSEK")

# Test 4 : Opération mathématique simple
print("\n4. Test calcul:")
try:
    result = np.sin(np.pi/2)
    print(f"   ✓ sin(π/2) = {result:.4f}")
except Exception as e:
    print(f"   ✗ ERREUR: {e}")

print("\n" + "="*70)
print("FIN DES TESTS")
print("="*70)