package kaldi

import (
	"fmt"
	"path"
	"strings"
)

type AmTask interface {
	Train() bool
	Decode(string) bool
	Score(string) ([][]string, error)
	DecodeSets([]string)
	ScoreSets([]string) [][]string
}

type GMM struct {
	Tag     string
	Target  string
	Align   string
	Feature string
	Dynamic string
}

func NewGMM() *GMM {
	return &GMM{"test", "gmm", "tri1", "bnf", ""}
}
func (g *GMM) TargeName() string {
	return JoinParams(g.Target, g.Tag)
}
func (g *GMM) TargetDir() string {
	return path.Join(NewPathConf(g.Feature).Exp, g.TargeName())
}
func (g *GMM) AlignDir() string {
	return path.Join(MfccConf().Exp,
		JoinParams(g.Align, "ali"))
}

func MkGraph(target_dir string) error {
	if !DirExist(target_dir) {
		Log("MkGraph Error: No Exist TargetDir:%s \n", target_dir)
		return fmt.Errorf("MkGraph Error: No Exist TargetDir:%s \n", target_dir)
	}
	Log("MkGraph of TargetDir:%s", target_dir)

	cmd_str := JoinArgs(
		"utils/mkgraph.sh",
		TestLang(),
		target_dir,
		Graph(target_dir))
	Log("CMD:%s", cmd_str)
	BashRun(cmd_str)
	return nil
}

func (g *GMM) DecodeDir(set string) string {
	return path.Join(g.TargetDir(), JoinParams("decode", "#"+path.Base(set)))
}

func (g *GMM) Train() {

	option := ""

	if g.Dynamic != "" {
		option += "--feat-type " + g.Dynamic
	}

	cmd_str := JoinArgs(
		"steps/train_deltas.sh",
		option,
		"2500 15000",
		FeatTrain(g.Feature),
		Lang(),
		g.AlignDir(),
		g.TargetDir(),
	)
	BashRun(cmd_str)
}
func (g *GMM) DecodeSets(sets []string) {
	for _, set := range sets {
		g.Decode(set)
	}
}

func (g *GMM) Decode(set string) bool {
	dirs, err := Subsets(set, g.Feature)
	if err != nil {
		fmt.Println(err)
		return false
	}
	for _, dir := range dirs {
		cmd_str := JoinArgs(
			"steps/decode.sh",
			"--nj 10",
			Graph(g.TargetDir()),
			path.Join(NewPathConf(g.Feature).FeatData, dir),
			g.DecodeDir(dir))
		BashRun(cmd_str)
	}

	return false
}

func (g *GMM) ScoreSets(datasets []string) [][]string {
	result := [][]string{}
	for _, set := range datasets {
		res, _ := g.Score(set)
		newslice := make([][]string, len(res)+len(result))
		copy(newslice, result)
		copy(newslice[len(result):], res)
		result = newslice
	}
	return result
}

func (g *GMM) Score(set string) ([][]string, error) {

	subsets, err := Subsets(set, g.Feature)
	if err != nil {
		fmt.Println(err)
		return [][]string{}, err
	}
	res_sets := [][]string{}
	for _, set := range subsets {
		decode_dir := g.DecodeDir(set)
		if !DirExist(decode_dir) {
			continue
		}

		cmd_str := "grep WER " + decode_dir + "/wer* | utils/best_wer.sh |awk '{print $2}'"
		output, _ := BashOutput(cmd_str)
		wer := strings.Trim(string(output[:len(output)]), " \n")
		modified_set := strings.Join(ExcludeDefault(Unique(set, true)), "_")
		res_sets = append(res_sets, []string{g.TargeName(), modified_set, wer})
	}
	return res_sets, nil
}
