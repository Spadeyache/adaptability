"""Synthetic collection guards; no simulator or new-cube generator is called."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from . import collection as c


class CollectionGuards(unittest.TestCase):
    def test_missing_selection_stops_before_random_draw(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c, 'ROOT', Path(directory)), \
             patch.object(c, 'verify_prepared'), patch.object(c.np.random, 'default_rng') as rng:
            with self.assertRaises(FileNotFoundError):
                c.initialize_test()
            rng.assert_not_called()

    def selection_fixture(self, root):
        results=root/'results';results.mkdir()
        c.write_new(results/'selected_config.json', {'synthetic':True})
        c.write_new(results/'selection_seal.json', {'files':{'results/selected_config.json':c.digest(results/'selected_config.json')}})
        c.write_new(results/'settings_announced.json', {
            'selected_config_sha256':c.digest(results/'selected_config.json'),
            'announced_unix_seconds':time.time()})
        return results

    def test_announced_selection_binds_frozen_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c,'ROOT',Path(directory)):
            root=Path(directory);results=self.selection_fixture(root)
            value=c.require_announced_selection()
            self.assertEqual(value['selected_config_sha256'],c.digest(results/'selected_config.json'))
            (results/'selected_config.json').write_text(json.dumps({'synthetic':'changed'}))
            with self.assertRaisesRegex(AssertionError,'Changed validation selection artifact'):
                c.require_announced_selection()

    def test_announcement_must_follow_selection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(c,'ROOT',Path(directory)):
            root=Path(directory);results=self.selection_fixture(root)
            value=c.read(results/'settings_announced.json');value['announced_unix_seconds']=0
            (results/'settings_announced.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(AssertionError,'predates'):
                c.require_announced_selection()


if __name__=='__main__':unittest.main()
