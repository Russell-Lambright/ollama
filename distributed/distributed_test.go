package distributed

import "testing"

func TestParseMode(t *testing.T) {
	cases := map[string]struct {
		in      string
		want    Mode
		wantErr bool
	}{
		"empty default":   {"", ModeStandalone, false},
		"standalone":      {"standalone", ModeStandalone, false},
		"primary":         {"primary", ModePrimary, false},
		"secondary":       {"secondary", ModeSecondary, false},
		"invalid":         {"worker", ModeStandalone, true},
		"case-sensitive":  {"Primary", ModeStandalone, true},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got, err := ParseMode(tc.in)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ParseMode(%q) err = %v, wantErr=%v", tc.in, err, tc.wantErr)
			}
			if got != tc.want {
				t.Fatalf("ParseMode(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestModeValid(t *testing.T) {
	for _, m := range []Mode{ModeStandalone, ModePrimary, ModeSecondary} {
		if !m.Valid() {
			t.Errorf("Mode(%q).Valid() = false, want true", m)
		}
	}
	if Mode("bogus").Valid() {
		t.Error(`Mode("bogus").Valid() = true, want false`)
	}
}

func TestModeString(t *testing.T) {
	if ModePrimary.String() != "primary" {
		t.Errorf("ModePrimary.String() = %q, want %q", ModePrimary.String(), "primary")
	}
}
