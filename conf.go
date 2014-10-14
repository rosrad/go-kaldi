package kaldi

import (
	"fmt"
	"github.com/codeskyblue/go-sh"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"
)

type PathBase struct {
	Exp       string
	FeatData  string
	FeatParam string
	Log       string
}

type PathConf struct {
	PathBase
	FeatExp  string
	FeatDump string
}

func RootPath() string {
	return "tmp"
}

func Lang() string {
	return path.Join(RootPath(), "data", "lang")
}

func TestLang() string {
	return path.Join(RootPath(), "data", "lang_test_bg_5k")
}

func Graph(target string) string {
	return path.Join(target, "graph_bg_5k")
}

func TrainData(pathconf *PathConf, cond string) string {
	train := "si_tr"
	if cond == "mc" {
		train = "REVERB_tr_cut/SimData_tr_for_1ch_A"
	}
	return path.Join(pathconf.FeatData, train)
}
func (c *PathConf) MixTag(mdl string) *PathConf {
	p := *c
	tag := "_bnf_" + mdl
	if tag != "" {
		p.Exp = path.Join(filepath.Dir(c.Exp), "exp"+tag)
		p.FeatParam = c.FeatParam + tag
		p.FeatData = c.FeatData + tag
	}
	return &p
}

func NewPathConf(feat string) *PathConf {
	return &PathConf{
		PathBase{
			Exp:       path.Join(RootPath(), "exp", feat),
			FeatData:  path.Join(RootPath(), "feats", feat, "data"),
			FeatParam: path.Join(RootPath(), "feats", feat, "param"),
			Log:       path.Join(RootPath(), "log", feat)},
		path.Join(RootPath(), "feats", feat, "exp"),
		path.Join(RootPath(), "feats", feat, "exp", "dump"),
	}
}

func (conf *PathConf) StoreName() map[string]string {
	names := make(map[string]string)
	names["exp"], _ = filepath.EvalSymlinks(conf.Exp)
	names["data"], _ = filepath.EvalSymlinks(conf.FeatData)
	names["param"], _ = filepath.EvalSymlinks(conf.FeatParam)
	return names
}

func (conf *PathConf) Bakup(dst_conf *PathConf) int {
	dst := Map2Slice(dst_conf.StoreName())
	src := Map2Slice(conf.StoreName())
	rename_num := 0
	for src_key, src_val := range src {
		if src_val == dst[src_key] && strings.Trim(src_val, " ") != "" {
			bakup := src_val + ".bakup." + time.Now().Format("20060102150405")
			fmt.Printf("Mv %s, %s \n", src_val, bakup)
			os.Rename(src_val, bakup)
			rename_num++
		}
	}
	return rename_num
}

func MfccConf() *PathConf {
	return NewPathConf("mfcc")
}

func BnfConf() *PathConf {
	return NewPathConf("bnf")
}

func MfccTrain() string {
	return TrainData(MfccConf(), "mc")
}

func BnfTrain() string {
	return TrainData(BnfConf(), "mc")
}

func FeatTrain(feature string) string {
	return TrainData(NewPathConf(feature), "mc")
}

func JoinParams(elem ...string) string {
	for i, e := range elem {
		if e != "" {
			return strings.Trim(strings.Join(elem[i:], "_"), "_")
		}
	}
	return ""
}

func JoinArgs(elem ...string) string {
	for i, e := range elem {
		if e != "" {
			return strings.Trim(strings.Join(elem[i:], " "), " ")
		}
	}
	return ""
}

func InsureDir(dir string) {
	if !DirExist(dir) {
		os.Mkdir(dir, os.ModePerm)
	}
}

func DirExist(dir string) bool {
	if src, err := os.Stat(dir); os.IsNotExist(err) || !src.IsDir() {
		return false
	}
	return true
}

func BashRun(cmd string) error {
	return sh.Command("bash", "-c", cmd).Run()
}

func BashOutput(cmd string) (out []byte, err error) {
	return sh.Command("bash", "-c", cmd).Output()
}

func Now() string {
	return time.Now().Format("2006-Jan-02")
}

func Log(format string, a ...interface{}) {
	fmt.Printf(format+"\n", a)
}

func Subsets(set, feature string) ([]string, error) {
	sets := []string{set}
	if !strings.HasPrefix(set, "si_") {
		sys_dirs, err := ioutil.ReadDir(path.Join(NewPathConf(feature).FeatData, set))
		if err != nil || len(sys_dirs) == 0 {
			return []string{}, fmt.Errorf("No Subset in %s", set)
		}
		sets = []string{}
		for _, tmp := range sys_dirs {
			sets = append(sets, path.Join(set, tmp.Name()))
		}
	}
	return sets, nil
}
func DataSets(dt_only bool) []string {
	if dt_only {
		return []string{"REVERB_dt", "PHONE_dt", "PHONE_SEL_dt", "PHONE_MLLD_dt"}
	}
	return []string{"REVERB_tr_cut", "REVERB_dt", "PHONE_dt", "PHONE_SEL_dt", "PHONE_MLLD_dt"}
}

func Contains(item string, set []string) bool {
	for _, value := range set {
		if value == item {
			return true
		}
	}
	return false
}

func Exclude(sets, blacklist []string) []string {
	modified := []string{}
	for _, item := range sets {
		if !Contains(item, blacklist) {
			modified = append(modified, item)
		}
	}
	return modified
}

func ExcludeDefault(set []string) []string {
	blacklist := []string{"dt", "for", "reverb", "simdata", "phone"}
	return Exclude(set, blacklist)
}

func Unique(set string, ignore_case bool) []string {
	var sub sort.StringSlice

	if ignore_case {
		set = strings.ToLower(set)
	}
	f := func(c rune) bool {
		return !unicode.IsLetter(c) && !unicode.IsNumber(c)
	}
	sub = strings.FieldsFunc(set, f)
	sub.Sort()
	prev := ""
	unique := []string{}
	for _, value := range sub {
		if prev != value {
			unique = append(unique, value)
		}

		prev = value
	}
	return unique
}

func FormatScore(result [][]string) string {
	format := ""
	split := "|-----------------------------|\n"
	format += split
	for j := 1; j < len(result[0]); j++ {
		format += fmt.Sprintf("|%s |", result[0][0])
		for i := 0; i < len(result); i++ {

			format += fmt.Sprintf("%s|", result[i][j])
		}
		format += "\n"
	}
	format += split
	return format
}

func FormatScoreSets(result map[string][][]string) string {
	format := ""
	split := "|-----------------------------|\n"
	for _, value := range result {
		format += split
		format += FormatScore(value)
	}
	format += split
	return format
}

func Map2Slice(src map[string]string) []string {
	dst := []string{}
	for _, value := range src {
		dst = append(dst, value)
	}
	sort.Sort(sort.StringSlice(dst))
	return dst
}
