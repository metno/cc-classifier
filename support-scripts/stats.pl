#!/usr/bin/perl -w

use strict;
use Data::Dumper;

my %cnt = ();
$cnt{'bigger_than_two'} = 0.0;
$cnt{'smaller_than_or_equal_two'} = 0.0;
$cnt{'smaller_than_or_equal_one'} = 0.0;
$cnt{'bigger_than_three'} = 0.0;
$cnt{'bigger_than_four'} = 0.0;
$cnt{'bigger_than_five'} = 0.0;
$cnt{'bigger_than_six'} = 0.0;
$cnt{'bigger_than_seven'} = 0.0;
$cnt{'equal'} = 0.0;
my $i = 0.0;

my %cnt2 = ();
for ( my $i = 0; $i <=8; $i++) {
    $cnt2{$i}{'bigger_than_two'} = 0.0;
    $cnt2{$i}{'smaller_than_or_equal_two'} = 0.0;
    $cnt2{$i}{'smaller_than_or_equal_one'} = 0.0;
    $cnt2{$i}{'bigger_than_three'} = 0.0;
    $cnt2{$i}{'bigger_than_four'} = 0.0;
    $cnt2{$i}{'bigger_than_five'} = 0.0;
    $cnt2{$i}{'bigger_than_six'} = 0.0;
    $cnt2{$i}{'bigger_than_seven'} = 0.0;
    $cnt2{$i}{'equal'} = 0.0;
}

my %lcounts = (0=>0, 1=>0, 2=>0, 3=>0, 4=>0, 5=>0, 6=>0, 7=>0, 8=>0, );

while (<>) {
    chomp;
    if ($_ =~ /(\S+)\s(\d)\s(\d)\s(\S+)/ ) {
	my $path = $1;
	my $cc = int($2);
	my $cc_cnn = int($3);
	my $spread = $4 + 0;
	
	#if ( $cc_cnn < 0 || $cc_cnn != 8) { # Bias ..
	if ( $cc_cnn < 0 ) {
	    next;
	} else {
	    $lcounts{$cc}++; 
	    $i++;

	    if ( abs($cc_cnn - $cc) > 3) {
		$cnt{'bigger_than_three'} = $cnt{'bigger_than_three'} + 1;
		$cnt2{$cc}{'bigger_than_three'} = $cnt2{$cc}{'bigger_than_three'} + 1;
	    }
	    if ( abs($cc_cnn - $cc) > 4) {
		$cnt{'bigger_than_four'} = $cnt{'bigger_than_four'} + 1;
		$cnt2{$cc}{'bigger_than_four'} = $cnt2{$cc}{'bigger_than_four'} + 1;
	    }
	    if ( abs($cc_cnn - $cc) > 5) {
		$cnt{'bigger_than_five'} = $cnt{'bigger_than_five'} + 1;
		$cnt2{$cc}{'bigger_than_five'} = $cnt2{$cc}{'bigger_than_five'} + 1;
	    }
	    if ( abs($cc_cnn - $cc) > 6) {
		$cnt{'bigger_than_six'} = $cnt{'bigger_than_six'} + 1;
		$cnt2{$cc}{'bigger_than_six'} = $cnt2{$cc}{'bigger_than_six'} + 1;
	    }
	    if ( abs($cc_cnn - $cc) > 7) {
		$cnt{'bigger_than_seven'} = $cnt{'bigger_than_seven'} + 1;
		$cnt2{$cc}{'bigger_than_seven'} = $cnt2{$cc}{'bigger_than_seven'} + 1;
	    }
	    if ( abs($cc_cnn - $cc) > 2) {
		print "$path HUH $cc $cc_cnn $spread\n";
		$cnt{'bigger_than_two'} = $cnt{'bigger_than_two'} + 1;
		$cnt2{$cc}{'bigger_than_two'} = $cnt2{$cc}{'bigger_than_two'} + 1;
	    }
	    if ( abs($cc_cnn - $cc) <= 2) {		
		$cnt{'smaller_than_or_equal_two'} = $cnt{'smaller_than_or_equal_two'} + 1;
		$cnt2{$cc}{'smaller_than_or_equal_two'} = $cnt2{$cc}{'smaller_than_or_equal_two'} + 1;
	    }
	    if ( abs($cc_cnn - $cc) <= 1) {
		$cnt{'smaller_than_or_equal_one'} = $cnt{'smaller_than_or_equal_one'} + 1;
		$cnt2{$cc}{'smaller_than_or_equal_one'} = $cnt2{$cc}{'smaller_than_or_equal_one'} + 1;
	    }
	    if ( abs($cc_cnn - $cc) == 0) {
		$cnt{'equal'} = $cnt{'equal'} + 1;
		$cnt2{$cc}{'equal'} = $cnt2{$cc}{'equal'} + 1;
	    }
	}
    }
}

print "Total: $i\n";
printf("Smaller than or equal to two %.2f%%\n",  ($cnt{'smaller_than_or_equal_two'} / $i) * 100);
printf("Smaller than or equal to one %.2f%%\n",  ($cnt{'smaller_than_or_equal_one'} / $i) * 100);
printf("Bigger than two %.2f%%\n",  ($cnt{'bigger_than_two'} / $i ) * 100);
printf("Equal: %.2f%%\n",  ($cnt{'equal'} / $i) * 100);

print("\nErrors <= 2 for each class:\n");
my $cnt = 0.0;
for ($i=0; $i<=8; $i++) {
    printf("$i: %0.2f%%\n",  100* ($cnt2{$i}{'smaller_than_or_equal_two'}/$lcounts{$i}));
    $cnt += ($cnt2{$i}{'smaller_than_or_equal_two'}/$lcounts{$i});
}

#print Dumper \%cnt2;
