import unittest
import os
import json
from src.user_manager import GestorUsuarios # Assuming this is the correct import path

class TestUserManagerPersistence(unittest.TestCase):
    """
    Test suite for UserManager persistence.
    """
    test_filepath = "test_temp_users.json"

    def setUp(self):
        # Ensure the test file does not exist before each test
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)

    def tearDown(self):
        # Clean up the test file after each test
        if os.path.exists(self.test_filepath):
            os.remove(self.test_filepath)

    def test_save_load_cycle(self):
        """
        Tests saving user data and then loading it into a new instance.
        """
        # --- First instance: Register user and save ---
        user_manager1 = GestorUsuarios(archivo=self.test_filepath)
        user_id = "persistent_user_123"
        expected_history_id = "historial_1" # Based on default counter start

        history_id1 = user_manager1.registrar_usuario(user_id)
        self.assertEqual(history_id1, expected_history_id)
        self.assertIn(user_id, user_manager1.usuarios)

        # registrar_usuario calls guardar() internally, so it should be saved.
        # user_manager1.guardar() # Explicit call if needed, but registrar_usuario should handle it

        # Verify file was created and contains the user
        self.assertTrue(os.path.exists(self.test_filepath))
        with open(self.test_filepath, 'r') as f:
            data_on_disk = json.load(f)
        self.assertIn(user_id, data_on_disk["usuarios"])
        self.assertEqual(data_on_disk["usuarios"][user_id], expected_history_id)
        self.assertEqual(data_on_disk["contador"], 1)

        del user_manager1 # Simulate shutdown/loss of instance

        # --- Second instance: Load users and verify ---
        user_manager2 = GestorUsuarios(archivo=self.test_filepath)
        # cargar() is called in __init__

        self.assertIn(user_id, user_manager2.usuarios, "User not found after loading.")
        history_id2 = user_manager2.obtener_historial(user_id)
        self.assertEqual(history_id2, expected_history_id, "History ID mismatch after loading.")
        self.assertEqual(user_manager2.contador, 1, "Counter mismatch after loading.")

        # Register another user to see if counter continues correctly
        user_id2 = "another_user_456"
        expected_history_id2 = "historial_2"
        history_id_new = user_manager2.registrar_usuario(user_id2)
        self.assertEqual(history_id_new, expected_history_id2)
        self.assertEqual(user_manager2.contador, 2)


if __name__ == '__main__':
    unittest.main()
