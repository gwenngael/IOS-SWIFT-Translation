IOS-SWIFT-Translation

A small Python script to bootstrap iOS localization from Swift source files.

What it does
	•	Scans .swift files recursively and extracts UI-facing string literals (SwiftUI + some UIKit cases).
	•	Generates Localizable.strings for English (en) as reference + fallback, plus: fr, de, es, it, nl.
	•	Optionally uses OpenAI to normalize keys and propose translations.
	•	Rewrites Swift sources:
	•	SwiftUI: replaces “Some text” with “some_key” (resolved via LocalizedStringKey)
	•	UIKit alerts: replaces “Some text” with NSLocalizedString(“some_key”, comment: “”)

Requirements
	•	Python 3.10+ recommended
	•	Dependencies: tqdm, openai, pydantic

Install:
pip install tqdm openai pydantic

OpenAI setup (optional)

Set your API key:
export OPENAI_API_KEY=“your_key_here”

Optional model override:
export OPENAI_MODEL=“gpt-4o-mini”

Usage

Dry-run (recommended first):
python firehook_l10n_bootstrap.py /path/to/YourXcodeProject –use-openai –limit 10

Apply (writes files + rewrites Swift):
python firehook_l10n_bootstrap.py /path/to/YourXcodeProject –use-openai –apply –backup

Output files (generated):
L10n/en.lproj/Localizable.strings
L10n/fr.lproj/Localizable.strings
L10n/de.lproj/Localizable.strings
L10n/es.lproj/Localizable.strings
L10n/it.lproj/Localizable.strings
L10n/nl.lproj/Localizable.strings

By default, en is filled and the other languages are written empty unless an existing translation already exists.

Fill non-English files with suggestions too:
python firehook_l10n_bootstrap.py /path/to/YourXcodeProject –use-openai –apply –backup –fill-suggestions

Xcode integration
	•	Add the generated .lproj folders to your Xcode project (File → Add Files to …).
	•	Ensure the correct target membership (app target + any extensions/widgets).

Notes / limitations
	•	Interpolated strings like “Hello (name)” are skipped (convert manually to localization-friendly patterns).
	•	URLs and code-like tokens are skipped.
	•	Strings that already look like localization keys are not modified.

License

MIT
